import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from tqdm import tqdm
from transformers import pipeline
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. INITIALIZATION
# ==========================================
API_KEY = 'PKH7VBANASKPY6VTXRL7HHTB63'
SECRET_KEY ='71pqaHS8VBYi5kzrFJ22DqUfJBUGLibk5NsGPzPkRMWz'
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)

SYMBOL = "NVDA"
YEARS_OF_DATA = 10

# Load FinBERT
print("Loading FinBERT Sentiment Analyzer...")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert", device=-1) # Change device=0 if you have a GPU

# ==========================================
# 2. FETCH HISTORICAL PRICE & TECHNICALS
# ==========================================
def build_technical_dataset():
    print(f"Fetching historical bars for {SYMBOL} ({YEARS_OF_DATA} years)...")
    start_date = datetime.now() - timedelta(days=365 * YEARS_OF_DATA)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Day,
        start=start_date
    )
    
    bars = data_client.get_stock_bars(request_params).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL, level="symbol")
        
    # 1. Moving Averages & Stationary Relative Ratios
    bars['SMA_10'] = bars['close'].rolling(window=10).mean()
    bars['SMA_50'] = bars['close'].rolling(window=50).mean()
    bars['SMA_200'] = bars['close'].rolling(window=200).mean()
    
    bars['SMA_10_Ratio'] = (bars['close'] / bars['SMA_10']) - 1.0
    bars['SMA_50_Ratio'] = (bars['close'] / bars['SMA_50']) - 1.0
    bars['SMA_200_Ratio'] = (bars['close'] / bars['SMA_200']) - 1.0

    # 2. Momentum: RSI (Normalized 0.0 to 1.0)
    change = bars['close'].diff()
    rsi_raw = 100 - (100 / (1 + (change.mask(change < 0, 0.0).ewm(com=13, adjust=False).mean() / 
                                        -change.mask(change > 0, 0.0).ewm(com=13, adjust=False).mean().replace(0, np.nan))))
    bars['RSI_14'] = rsi_raw.fillna(50) / 100.0

    # 3. Volatility: ATR 14 & Percentage Volatility (ATR_Pct - Stationary)
    high_low = bars['high'] - bars['low']
    high_close = (bars['high'] - bars['close'].shift()).abs()
    low_close = (bars['low'] - bars['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    bars['ATR_14'] = true_range.rolling(14).mean()
    bars['ATR_Pct'] = bars['ATR_14'] / bars['close']

    # 4. Volatility: Bollinger Bands (%B)
    sma_20 = bars['close'].rolling(20).mean()
    std_20 = bars['close'].rolling(20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bars['BB_Pct'] = (bars['close'] - bb_lower) / ((bb_upper - bb_lower).replace(0, np.nan))
    bars['BB_Pct'] = bars['BB_Pct'].fillna(0.5)

    # 5. Volume: VWAP Ratio (Percentage Deviation)
    bars['VWAP_Ratio'] = (bars['close'] / bars['vwap']) - 1.0

    # 6. Volume: On-Balance Volume (OBV 20-day Z-Score)
    obv_change = np.sign(bars['close'].diff()).fillna(0)
    obv = (obv_change * bars['volume']).fillna(0).cumsum()
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std().replace(0, np.nan)
    bars['OBV_Z'] = ((obv - obv_mean) / obv_std).fillna(0)
    
    # Target Variable: 1 if next day closes higher, 0 if lower
    bars['Target'] = (bars['close'].shift(-1) > bars['close']).astype(int)
    
    return bars.dropna()

# ==========================================
# 3. FETCH & SCORE HISTORICAL SENTIMENT
# ==========================================
def build_sentiment_dataset(dates):
    csv_file = f"{SYMBOL}_historical_features.csv"
    try:
        existing_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        if 'Sentiment_Score' in existing_df.columns and len(existing_df) > 0:
            print("Found existing cached Sentiment_Score. Reusing cached news sentiment...")
            existing_df.index = pd.to_datetime(existing_df.index)
            sentiment_df = existing_df[['Sentiment_Score']].reindex(pd.to_datetime(dates), fill_value=0.0)
            sentiment_df.index.name = 'Date'
            return sentiment_df
    except Exception as e:
        print(f"Could not load cached sentiment: {e}")

    print("Fetching and scoring historical news. This will take a while...")
    sentiment_records = []
    
    # Iterate through unique dates in the technical dataset
    for date in tqdm(dates, desc="Processing Daily Sentiment"):
        # Create a 24-hour window for each trading day
        start_time = pd.to_datetime(date).replace(hour=0, minute=0, second=0)
        end_time = start_time + timedelta(days=1)
        
        try:
            news_req = NewsRequest(
                symbols=SYMBOL, 
                start=start_time, 
                end=end_time, 
                limit=15
            )
            news_response = news_client.get_news(news_req)
            
            if hasattr(news_response, 'data'):
                articles = news_response.data
            elif hasattr(news_response, 'news'):
                articles = news_response.news
            else:
                articles = []

            if not articles:
                sentiment_records.append({'Date': date, 'Sentiment_Score': 0.0})
                continue

            headlines = " ".join([article.headline for article in articles])
            
            # FinBERT limits input size to 512 tokens. Truncate if necessary.
            sentiment_result = sentiment_analyzer(headlines[:1500])[0] 
            
            if sentiment_result['label'] == 'positive':
                score = sentiment_result['score']
            elif sentiment_result['label'] == 'negative':
                score = -sentiment_result['score']
            else:
                score = 0.0
                
            sentiment_records.append({'Date': date, 'Sentiment_Score': score})
            
        except Exception as e:
            sentiment_records.append({'Date': date, 'Sentiment_Score': 0.0})
            
    sentiment_df = pd.DataFrame(sentiment_records)
    sentiment_df.set_index('Date', inplace=True)
    return sentiment_df

# ==========================================
# 4. EXECUTE TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Build Base Dataset
    df_tech = build_technical_dataset()
    
    # 2. Build Sentiment Dataset
    # We pass the index (dates) from the technical dataset to align them perfectly
    df_sentiment = build_sentiment_dataset(df_tech.index)
    
    # 3. Merge Datasets
    final_df = df_tech.join(df_sentiment)
    final_df['Sentiment_Score'] = final_df['Sentiment_Score'].fillna(0.0)
    
    # Save to CSV so you don't have to re-run FinBERT during model tweaking
    final_df.to_csv(f"{SYMBOL}_historical_features.csv")
    print(f"\nData saved to {SYMBOL}_historical_features.csv")

    # 4. Train XGBoost with stationary technical, volatility, volume, and sentiment features
    features = [
        'SMA_10_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 
        'RSI_14', 'ATR_Pct', 'BB_Pct', 'VWAP_Ratio', 'OBV_Z', 'Sentiment_Score'
    ]
    
    X = final_df[features]
    y = final_df['Target']

    # Sequential split (no shuffling) to prevent look-ahead bias
    split_idx = int(len(final_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTraining XGBoost model on {len(X_train)} days of data...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Test Accuracy: {accuracy * 100:.2f}%")

    # 5. Save the updated model
    model.save_model("nvda_xgboost_model.json")
    print("New model saved as 'nvda_xgboost_model.json'. Ready for live deployment.")