"""Daily order proposals; paper submission is an explicit opt-in."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
from strategy import FEATURES, prepare, plan, black_scholes
from model_builder import probability

ROOT=Path(__file__).resolve().parent


def propose(bars,bundle,equity=10000.):
    d=prepare(bars)
    if len(d)<200 or not np.isfinite(d.iloc[-1][FEATURES].astype(float)).all():
        raise ValueError('Need at least 200 completed candles and valid indicators')
    p=float(probability(bundle,d.iloc[[-1]])[0])
    decision=plan(d,p,bundle['threshold'],equity,bundle['horizon']).to_dict()
    if not bundle['enabled']:
        decision.update(action='WAIT',reason='Validation did not qualify any confidence threshold',
                        stop=None,target=None,quantity=0)
    row=d.iloc[-1]
    decision['structure']={'support':float(row.support),'resistance':float(row.resistance),
                           'candlestick_score':int(row.candle),'fair_value_gap_direction':int(row.fvg),
                           'adx':float(row.adx)}
    decision['black_scholes_atm_scenario']=black_scholes(float(row.close),float(row.close),
                                                        bundle['horizon']/252,float(row.volatility))
    decision['signal_bar']=str(d.index[-1])
    decision['horizon_sessions']=bundle['horizon']
    return decision


def fetch_bars(key,secret):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    now=datetime.now(timezone.utc)
    client=StockHistoricalDataClient(key,secret)
    request=StockBarsRequest(symbol_or_symbols=['NVDA'],timeframe=TimeFrame.Day,
                             start=now-timedelta(days=800),adjustment='all',feed='iex')
    bars=client.get_stock_bars(request).df.xs('NVDA',level='symbol')
    # Conservative: discard the entire current New York calendar date, including
    # a possibly incomplete daily candle, regardless of host timezone.
    dates=bars.index.tz_convert('America/New_York').date
    today=pd.Timestamp(now).tz_convert('America/New_York').date()
    return bars.loc[dates<today].sort_index()



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/strategy.joblib')
    p.add_argument('--data',type=Path,help='Offline CSV; prints a historical proposal only')
    p.add_argument('--equity',type=float,default=10000.)
    p.add_argument('--submit-paper',action='store_true',help='Explicitly submit an Alpaca paper bracket')
    a=p.parse_args()
    if a.submit_paper and a.data: p.error('Paper submission requires fresh Alpaca data, not a CSV')
    bundle=joblib.load(a.bundle)
    key,secret=os.environ.get('ALPACA_API_KEY'),os.environ.get('ALPACA_SECRET_KEY')
    if a.data:
        bars=pd.read_csv(a.data,index_col=0,parse_dates=True).sort_index()
    else:
        if not key or not secret: p.error('Set ALPACA_API_KEY and ALPACA_SECRET_KEY securely, or use --data')
        bars=fetch_bars(key,secret)
        now=pd.Timestamp.now(tz='UTC')
        if bars.empty or now-bars.index[-1]>pd.Timedelta(days=5):
            raise RuntimeError('Market data is stale; no proposal produced')
    decision=propose(bars,bundle,a.equity)
    print(json.dumps(decision,indent=2,allow_nan=False))
    if a.submit_paper:
        from execution import execute
        print(execute(decision,key,secret,float(bars.close.iloc[-1]),bundle['horizon']))
