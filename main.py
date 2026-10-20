import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import pipeline

# Alpaca Trading & Orders
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

# Alpaca Market Data & News
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. INITIALIZATION
# ==========================================
API_KEY = 'PKH7VBANASKPY6VTXRL7HHTB63'
SECRET_KEY ='71pqaHS8VBYi5kzrFJ22DqUfJBUGLibk5NsGPzPkRMWz'

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)

SYMBOL = "NVDA"
QTY = 10

# The 9 stationary features your model was trained on
FEATURES = [
    'SMA_10_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 
    'RSI_14', 'ATR_Pct', 'BB_Pct', 'VWAP_Ratio', 'OBV_Z', 'Sentiment_Score'
]

print("Loading XGBoost model...")
model = xgb.XGBClassifier()
model.load_model("nvda_xgboost_model.json")

print("Loading FinBERT Sentiment Analyzer...")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

# ==========================================
# 2. FEATURE ENGINEERING (LIVE DATA)
# ==========================================
def get_live_features():
    print(f"[{datetime.now()}] Fetching live market data for {SYMBOL}...")
    
    # Request 350 days to ensure SMA_200 has full history
    request_params = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - pd.Timedelta(days=350)
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

    # 3. Volatility: ATR 14 & Percentage Volatility (ATR_Pct)
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

    current_state = bars.iloc[[-1]].copy()
    
    current_price = float(current_state['close'].values[0])
    current_atr = float(current_state['ATR_14'].values[0]) if not pd.isna(current_state['ATR_14'].values[0]) else (current_price * 0.02)
    sma_200_val = float(current_state['SMA_200'].values[0]) if not pd.isna(current_state['SMA_200'].values[0]) else current_price
    is_bull_market = current_price >= sma_200_val

    # --- NEWS SENTIMENT (FinBERT) ---
    news_req = NewsRequest(symbols=SYMBOL, limit=5)
    try:
        news_response = news_client.get_news(news_req)
        
        raw_articles = []
        if hasattr(news_response, 'news') and isinstance(news_response.news, list):
            raw_articles = news_response.news
        elif hasattr(news_response, 'data') and isinstance(news_response.data, dict):
            for val in news_response.data.values():
                if isinstance(val, list):
                    raw_articles.extend(val)
                    
        headlines_list = []
        for article in raw_articles:
            if hasattr(article, 'headline'):
                headlines_list.append(article.headline)
            elif isinstance(article, dict) and 'headline' in article:
                headlines_list.append(article['headline'])
                
        headlines = " ".join(headlines_list)
        
        if not headlines:
            sentiment_score = 0.0
        else:
            sentiment_result = sentiment_analyzer(headlines[:1500])[0] 
            if sentiment_result['label'] == 'positive':
                sentiment_score = sentiment_result['score']
            elif sentiment_result['label'] == 'negative':
                sentiment_score = -sentiment_result['score']
            else:
                sentiment_score = 0.0 
                
    except Exception as e:
        print(f"Could not fetch/score news: {e}")
        sentiment_score = 0.0 

    current_state['Sentiment_Score'] = sentiment_score
    
    return current_state[FEATURES], current_price, current_atr, is_bull_market

def close_position_safely(symbol):
    """
    Cancels all open orders for a symbol, waits until Alpaca clears them,
    and liquidates the position without triggering race conditions.
    """
    open_orders = trading_client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    for order in open_orders:
        try:
            trading_client.cancel_order_by_id(order.id)
        except Exception:
            pass
            
    print("Waiting for open/bracket orders to finish canceling...")
    while True:
        remaining_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        if not remaining_orders:
            break
        time.sleep(1)

    print("Liquidating position...")
    while True:
        try:
            trading_client.close_position(symbol)
            break
        except APIError as e:
            err_msg = str(e).lower()
            if any(code in err_msg for code in ["40310000", "42210000", "insufficient qty", "pending cancel"]):
                print("Alpaca processing order state change... Retrying in 2 seconds.")
                time.sleep(2)
            else:
                raise e

    print("Confirming position liquidation...")
    while True:
        active_positions = trading_client.get_all_positions()
        if not any(p.symbol == symbol for p in active_positions):
            break
        time.sleep(1)
        
    print("Position cleanly closed.")

# ==========================================
# 3. EXECUTION LOGIC (DYNAMIC ATR BRACKETS & TREND FILTER)
# ==========================================
def execute_trade():
    try:
        positions = trading_client.get_all_positions()
        current_pos = next((p for p in positions if p.symbol == SYMBOL), None)
        
        position_side = None
        if current_pos:
            position_side = current_pos.side.value.lower() if hasattr(current_pos.side, 'value') else str(current_pos.side).lower()

        features_df, current_price, current_atr, is_bull_market = get_live_features()
        prediction = model.predict(features_df)[0] 
        
        print(f"[{datetime.now()}] Signal: {'BUY/LONG' if prediction == 1 else 'SELL/SHORT'} | Price: ${current_price:.2f} | ATR: ${current_atr:.2f} | Bull Regime: {is_bull_market} | Position: {position_side or 'NONE'}")

        # --------------------------------------------------
        # SIGNAL: BULLISH (1) AND BULL REGIME -> Enter or Hold LONG
        # --------------------------------------------------
        if prediction == 1 and is_bull_market:
            if position_side == 'short':
                print("Signal changed to LONG. Closing SHORT position...")
                close_position_safely(SYMBOL)
                position_side = None
            if position_side is None:
                print("Action: EXECUTING LONG (ATR Dynamic Bracket)")
                take_profit_price = round(current_price + (2.0 * current_atr), 2)
                stop_loss_price = round(current_price - (1.0 * current_atr), 2)

                long_order = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=QTY,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price)
                )
                trading_client.submit_order(order_data=long_order)
                print(f"Long Order Placed! TP: ${take_profit_price} | SL: ${stop_loss_price}")
            else:
                print("Action: HOLD LONG")

        # --------------------------------------------------
        # SIGNAL: BEARISH (0) AND BEAR REGIME -> Enter or Hold SHORT
        # --------------------------------------------------
        elif prediction == 0 and not is_bull_market:
            if position_side == 'long':
                print("Signal changed to SHORT. Closing LONG position...")
                close_position_safely(SYMBOL)
                position_side = None
            if position_side is None:
                print("Action: EXECUTING SHORT (ATR Dynamic Bracket)")
                take_profit_price = round(current_price - (2.0 * current_atr), 2)
                stop_loss_price = round(current_price + (1.0 * current_atr), 2)

                short_order = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=QTY,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price)
                )
                trading_client.submit_order(order_data=short_order)
                print(f"Short Order Placed! TP: ${take_profit_price} | SL: ${stop_loss_price}")
            else:
                print("Action: HOLD SHORT")

        else:
            if position_side is not None:
                print("Signal conflicts with 200-day secular trend. Closing active position safely...")
                close_position_safely(SYMBOL)
            else:
                print("Action: NO TRADE (Filtered out by 200-day Secular Trend Filter)")

    except Exception as e:
        print(f"Trading Execution Error: {e}")

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Starting Live Trading Bot for {SYMBOL}...")
    while True:
        execute_trade()
        time.sleep(86400)

