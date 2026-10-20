import time
import functools
from alpaca.common.exceptions import APIError

def with_retry(max_retries=3, initial_delay=2.0, backoff_factor=2.0, exceptions=(APIError, Exception)):
    """
    Decorator for wrapping network/API calls with exponential backoff retry logic.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    err_str = str(e)
                    print(f"[Retry Warning] Attempt {attempt}/{max_retries} for '{func.__name__}' failed: {err_str}")
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
            print(f"[Retry Error] All {max_retries} attempts failed for '{func.__name__}'. Raising final exception.")
            raise last_exception
        return wrapper
    return decorator

def calculate_position_size(portfolio_equity, current_price, atr, risk_pct=0.01, sl_atr_mult=1.5):
    """
    Calculates volatility-adjusted position size based on portfolio equity, ATR volatility, and risk percentage.
    
    Formula:
      Dollar Risk = Equity * risk_pct (e.g. $100,000 * 1% = $1,000)
      Stop Loss Distance per Share = sl_atr_mult * ATR (e.g. 1.5 * $4.00 = $6.00)
      Position Shares (Qty) = int(Dollar Risk / Stop Distance)
    """
    if portfolio_equity <= 0 or current_price <= 0 or atr <= 0:
        return 1, 0.0, 0.0
        
    dollar_risk = portfolio_equity * risk_pct
    stop_distance = max(0.01, sl_atr_mult * atr)
    
    raw_qty = int(dollar_risk / stop_distance)
    
    # Cap position value to 95% of total portfolio equity to prevent over-leverage
    max_affordable_qty = int((portfolio_equity * 0.95) / current_price)
    
    qty = max(1, min(raw_qty, max_affordable_qty if max_affordable_qty > 0 else 1))
    
    return qty, dollar_risk, stop_distance

if __name__ == "__main__":
    # Example Test: Portfolio Equity $100,000, NVDA @ $120.00, ATR = $4.00 (Low Volatility)
    qty1, risk1, dist1 = calculate_position_size(100000, 120.0, 4.0, risk_pct=0.01, sl_atr_mult=1.5)
    print(f"Low Volatility (ATR=$4.00): Buy {qty1} shares | Dollar Risk: ${risk1:.2f} | Stop Dist: ${dist1:.2f}")

    # Example Test: Portfolio Equity $100,000, NVDA @ $120.00, ATR = $10.00 (High Volatility)
    qty2, risk2, dist2 = calculate_position_size(100000, 120.0, 10.0, risk_pct=0.01, sl_atr_mult=1.5)
    print(f"High Volatility (ATR=$10.00): Buy {qty2} shares | Dollar Risk: ${risk2:.2f} | Stop Dist: ${dist2:.2f}")
