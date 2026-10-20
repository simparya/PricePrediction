import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import xgboost as xgb

import db_manager
from config import get_alpaca_credentials
from notifier import send_discord_alert, COLOR_BUY, COLOR_SELL, COLOR_INFO, COLOR_WARNING
from utils import with_retry, calculate_position_size

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
API_KEY, SECRET_KEY = get_alpaca_credentials()

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)

SYMBOL = "NVDA"
CONFIDENCE_THRESHOLD = 0.70  # Model confidence must strictly be >= 70% to trade
RISK_PER_TRADE_PCT = 0.01    # Risk 1.0% of portfolio equity per trade
SL_ATR_MULT = 1.5            # Dynamic 1.5x ATR Stop Loss
TP_ATR_MULT = 3.0            # Dynamic 3.0x ATR Take Profit (2:1 Reward to Risk Ratio)

# Initialize SQLite Audit Log
db_manager.init_db()

# The 9 stationary features your model was trained on
FEATURES = [
    'SMA_10_Ratio', 'SMA_50_Ratio', 'SMA_200_Ratio', 
    'RSI_14', 'ATR_Pct', 'BB_Pct', 'VWAP_Ratio', 'OBV_Z', 'Sentiment_Score'
]

print("Loading XGBoost model...")
model = xgb.XGBClassifier()
model.load_model("nvda_xgboost_model.json")

_sentiment_analyzer = None

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from transformers import pipeline
            print("Loading FinBERT Sentiment Analyzer...")
            _sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Warning: Could not load FinBERT sentiment analyzer due to environment/DLL restriction: {e}")
            _sentiment_analyzer = False
    return _sentiment_analyzer if _sentiment_analyzer is not False else None

@with_retry(max_retries=3)
def fetch_stock_bars(request_params):
    return data_client.get_stock_bars(request_params).df

@with_retry(max_retries=3)
def get_account_equity():
    account = trading_client.get_account()
    return float(account.equity)

@with_retry(max_retries=3)
def submit_order_safe(order_data):
    return trading_client.submit_order(order_data=order_data)

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
    
    bars = fetch_stock_bars(request_params)
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
            analyzer = get_sentiment_analyzer()
            if analyzer is not None:
                sentiment_result = analyzer(headlines[:1500])[0] 
                if sentiment_result['label'] == 'positive':
                    sentiment_score = sentiment_result['score']
                elif sentiment_result['label'] == 'negative':
                    sentiment_score = -sentiment_result['score']
                else:
                    sentiment_score = 0.0 
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
        probabilities = model.predict_proba(features_df)[0]
        prob_down, prob_up = probabilities[0], probabilities[1]
        confidence = max(prob_up, prob_down)
        
        # Calculate Volatility-Adjusted Position Size based on 1% Equity Risk & ATR
        try:
            equity = get_account_equity()
        except Exception:
            equity = 100000.0  # Fallback paper capital
            
        qty, dollar_risk, stop_dist = calculate_position_size(
            portfolio_equity=equity,
            current_price=current_price,
            atr=current_atr,
            risk_pct=RISK_PER_TRADE_PCT,
            sl_atr_mult=SL_ATR_MULT
        )
        
        print(f"[{datetime.now()}] Model Confidence: UP={prob_up*100:.1f}%, DOWN={prob_down*100:.1f}% | Price: ${current_price:.2f} | ATR: ${current_atr:.2f} | Equity: ${equity:,.2f} | Risked Shares: {qty} | Bull Regime: {is_bull_market} | Position: {position_side or 'NONE'}")

        # Determine signal action string for SQLite logging
        if prob_up >= CONFIDENCE_THRESHOLD and is_bull_market:
            action_str = "BUY"
        elif prob_down >= CONFIDENCE_THRESHOLD and not is_bull_market:
            action_str = "SELL"
        else:
            action_str = "HOLD/NO_TRADE"

        # Log prediction to SQLite Database Audit Log
        db_manager.log_prediction(
            symbol=SYMBOL,
            close_price=current_price,
            atr=current_atr,
            prob_up=prob_up,
            prob_down=prob_down,
            confidence=confidence,
            is_bull_market=is_bull_market,
            action=action_str
        )

        # --------------------------------------------------
        # SIGNAL: BULLISH (prob_up >= 70%) AND BULL REGIME -> Enter or Hold LONG
        # --------------------------------------------------
        if prob_up >= CONFIDENCE_THRESHOLD and is_bull_market:
            if position_side == 'short':
                print("Signal changed to LONG. Closing SHORT position...")
                close_position_safely(SYMBOL)
                position_side = None
            if position_side is None:
                take_profit_price = round(current_price + (TP_ATR_MULT * current_atr), 2)
                stop_loss_price = round(current_price - (SL_ATR_MULT * current_atr), 2)
                
                print(f"Action: EXECUTING LONG (Confidence: {prob_up*100:.1f}% >= {CONFIDENCE_THRESHOLD*100:.0f}% | Qty: {qty} shares | TP: ${take_profit_price} | SL: ${stop_loss_price})")

                long_order = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price)
                )
                
                order_res = submit_order_safe(long_order)
                order_id = getattr(order_res, 'id', '')
                
                # Log trade execution to SQLite
                db_manager.log_trade(
                    symbol=SYMBOL,
                    side="BUY",
                    qty=qty,
                    entry_price=current_price,
                    sl_price=stop_loss_price,
                    tp_price=take_profit_price,
                    confidence=prob_up,
                    status="SUBMITTED",
                    order_id=order_id
                )
                
                # Dispatch Discord Alert Notification
                send_discord_alert(
                    title=f"🚀 EXECUTED BUY LONG: {SYMBOL}",
                    description=f"Automated trade submitted with 1% equity risk and dynamic ATR bracket.",
                    color=COLOR_BUY,
                    fields=[
                        {"name": "Symbol", "value": SYMBOL, "inline": True},
                        {"name": "Quantity", "value": str(qty), "inline": True},
                        {"name": "Entry Price", "value": f"${current_price:.2f}", "inline": True},
                        {"name": "AI Confidence", "value": f"{prob_up*100:.1f}%", "inline": True},
                        {"name": "Take Profit", "value": f"${take_profit_price:.2f} (+{TP_ATR_MULT}x ATR)", "inline": True},
                        {"name": "Stop Loss", "value": f"${stop_loss_price:.2f} (-{SL_ATR_MULT}x ATR)", "inline": True}
                    ]
                )
                print(f"Long Order Placed! Order ID: {order_id} | TP: ${take_profit_price} | SL: ${stop_loss_price}")
            else:
                print("Action: HOLD LONG")

        # --------------------------------------------------
        # SIGNAL: BEARISH (prob_down >= 70%) AND BEAR REGIME -> Enter or Hold SHORT
        # --------------------------------------------------
        elif prob_down >= CONFIDENCE_THRESHOLD and not is_bull_market:
            if position_side == 'long':
                print("Signal changed to SHORT. Closing LONG position...")
                close_position_safely(SYMBOL)
                position_side = None
            if position_side is None:
                take_profit_price = round(current_price - (TP_ATR_MULT * current_atr), 2)
                stop_loss_price = round(current_price + (SL_ATR_MULT * current_atr), 2)

                print(f"Action: EXECUTING SHORT (Confidence: {prob_down*100:.1f}% >= {CONFIDENCE_THRESHOLD*100:.0f}% | Qty: {qty} shares | TP: ${take_profit_price} | SL: ${stop_loss_price})")

                short_order = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price)
                )
                
                order_res = submit_order_safe(short_order)
                order_id = getattr(order_res, 'id', '')
                
                # Log trade execution to SQLite
                db_manager.log_trade(
                    symbol=SYMBOL,
                    side="SELL",
                    qty=qty,
                    entry_price=current_price,
                    sl_price=stop_loss_price,
                    tp_price=take_profit_price,
                    confidence=prob_down,
                    status="SUBMITTED",
                    order_id=order_id
                )
                
                # Dispatch Discord Alert Notification
                send_discord_alert(
                    title=f"📉 EXECUTED SELL SHORT: {SYMBOL}",
                    description=f"Automated short trade submitted with 1% equity risk and dynamic ATR bracket.",
                    color=COLOR_SELL,
                    fields=[
                        {"name": "Symbol", "value": SYMBOL, "inline": True},
                        {"name": "Quantity", "value": str(qty), "inline": True},
                        {"name": "Entry Price", "value": f"${current_price:.2f}", "inline": True},
                        {"name": "AI Confidence", "value": f"{prob_down*100:.1f}%", "inline": True},
                        {"name": "Take Profit", "value": f"${take_profit_price:.2f} (-{TP_ATR_MULT}x ATR)", "inline": True},
                        {"name": "Stop Loss", "value": f"${stop_loss_price:.2f} (+{SL_ATR_MULT}x ATR)", "inline": True}
                    ]
                )
                print(f"Short Order Placed! Order ID: {order_id} | TP: ${take_profit_price} | SL: ${stop_loss_price}")
            else:
                print("Action: HOLD SHORT")

        else:
            if position_side is not None:
                print(f"Confidence (UP={prob_up*100:.1f}%, DOWN={prob_down*100:.1f}%) below {CONFIDENCE_THRESHOLD*100:.0f}% threshold or conflicts with secular trend. Closing active position safely...")
                close_position_safely(SYMBOL)
                send_discord_alert(
                    title=f"⚠️ LIQUIDATED POSITION: {SYMBOL}",
                    description=f"Active position closed due to trend mismatch or confidence decay below threshold.",
                    color=COLOR_WARNING
                )
            else:
                print(f"Action: NO TRADE (Confidence UP={prob_up*100:.1f}%, DOWN={prob_down*100:.1f}% did not meet strictly >= {CONFIDENCE_THRESHOLD*100:.0f}% threshold or regime filter)")

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
