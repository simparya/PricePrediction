import pandas as pd
import numpy as np
import xgboost as xgb
from config import get_alpaca_credentials
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. INITIALIZATION
# ==========================================
API_KEY, SECRET_KEY = get_alpaca_credentials()
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)

SYMBOL = "NVDA"
YEARS_OF_DATA = 10

# Lazy loading helper for FinBERT
_sentiment_analyzer = None

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from transformers import pipeline
            print("Loading FinBERT Sentiment Analyzer...")
            _sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert", device=-1)
        except Exception as e:
            print(f"Warning: Could not load FinBERT sentiment analyzer due to environment/DLL restriction: {e}")
            _sentiment_analyzer = False
    return _sentiment_analyzer if _sentiment_analyzer is not False else None

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
            
            analyzer = get_sentiment_analyzer()
            if analyzer is not None:
                # FinBERT limits input size to 512 tokens. Truncate if necessary.
                sentiment_result = analyzer(headlines[:1500])[0] 
                
                if sentiment_result['label'] == 'positive':
                    score = sentiment_result['score']
                elif sentiment_result['label'] == 'negative':
                    score = -sentiment_result['score']
                else:
                    score = 0.0
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

    # 4. Extract Features & Target
    features = [
        'SMA_10_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 
        'RSI_14', 'ATR_Pct', 'BB_Pct', 'VWAP_Ratio', 'OBV_Z', 'Sentiment_Score'
    ]
    
    X = final_df[features]
    y = final_df['Target']

    print(f"\n==========================================")
    print("PHASE 2: WALK-FORWARD HYPERPARAMETER TUNING")
    print("==========================================")
    print(f"Dataset Size: {len(X)} trading days")
    
    # 5. Define Time-Series Cross Validation & Hyperparameter Grid
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    print("Running GridSearchCV with 5-Fold Time-Series Cross Validation...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    print("\nMathematically Optimal Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # 6. Evaluate Walk-Forward Performance across Folds using Optimal Parameters
    print("\n==========================================")
    print("WALK-FORWARD TIME-SERIES CROSS VALIDATION RESULTS")
    print("==========================================")
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        fold_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        fold_model.fit(X_tr, y_tr)
        
        val_preds = fold_model.predict(X_val)
        val_probs = fold_model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        auc = roc_auc_score(y_val, val_probs)
        
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_aucs.append(auc)
        
        train_start = X.index[train_idx[0]].strftime('%Y-%m-%d')
        train_end = X.index[train_idx[-1]].strftime('%Y-%m-%d')
        val_start = X.index[val_idx[0]].strftime('%Y-%m-%d')
        val_end = X.index[val_idx[-1]].strftime('%Y-%m-%d')
        
        print(f"Fold {fold}: Train [{train_start} -> {train_end}] | Val [{val_start} -> {val_end}] | Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")

    print(f"\nOverall Time-Series CV Accuracy: {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f}%)")
    print(f"Overall Time-Series CV ROC-AUC:  {np.mean(fold_aucs):.4f}")

    # 7. Retrain Best Model on Complete Dataset & Save
    best_model = grid_search.best_estimator_
    best_model.save_model("nvda_xgboost_model.json")
    print("\nTuned model saved as 'nvda_xgboost_model.json'. Ready for live deployment.")