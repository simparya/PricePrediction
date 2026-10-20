import sqlite3
import os
from datetime import datetime
from pathlib import Path
from .paths import ARTIFACTS_DIR

DB_FILE = str(ARTIFACTS_DIR / "trading_bot.db")

def init_db(db_path=DB_FILE):
    """Initializes SQLite database tables for audit logging."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table 1: Model Predictions & Market Features Audit Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        close_price REAL NOT NULL,
        atr REAL NOT NULL,
        prob_up REAL NOT NULL,
        prob_down REAL NOT NULL,
        confidence REAL NOT NULL,
        is_bull_market INTEGER NOT NULL,
        action TEXT NOT NULL
    )
    """)
    
    # Table 2: Trade Executions Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        confidence REAL NOT NULL,
        status TEXT NOT NULL,
        alpaca_order_id TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    return db_path

def log_prediction(symbol, close_price, atr, prob_up, prob_down, confidence, is_bull_market, action, db_path=DB_FILE):
    """Logs a live model prediction and feature state to the database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now_str = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO predictions (timestamp, symbol, close_price, atr, prob_up, prob_down, confidence, is_bull_market, action)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, symbol, float(close_price), float(atr), float(prob_up), float(prob_down), float(confidence), 1 if is_bull_market else 0, action))
    
    conn.commit()
    conn.close()

def log_trade(symbol, side, qty, entry_price, sl_price, tp_price, confidence, status="SUBMITTED", order_id=None, db_path=DB_FILE):
    """Logs a trade execution order to the database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now_str = datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO trades (timestamp, symbol, side, qty, entry_price, sl_price, tp_price, confidence, status, alpaca_order_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, symbol, side, int(qty), float(entry_price), float(sl_price), float(tp_price), float(confidence), status, str(order_id or "")))
    
    conn.commit()
    conn.close()

def get_recent_predictions(limit=5, db_path=DB_FILE):
    """Retrieves recent prediction audit records."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    db = init_db()
    print(f"SQLite Database initialized successfully at '{db}'.")
