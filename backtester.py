"""Held-out, next-open execution of the shared strategy. No browser side effects."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from strategy import FEATURES, prepare, plan
from model_builder import probability

ROOT=Path(__file__).resolve().parent


def run(data, bundle_path, output):
    bundle=joblib.load(bundle_path)  # Load only locally trained/trusted artifacts.
    d=prepare(pd.read_csv(data,index_col=0,parse_dates=True).sort_index())
    valid=d[FEATURES].notna().all(axis=1)
    d['probability']=np.nan
    d.loc[valid,'probability']=probability(bundle,d.loc[valid])
    # Backtesting engine receives warmup too, but may trade only after test_start.
    test_start=pd.Timestamp(bundle['test_start'])
    horizon=bundle['horizon']
    last_entry=len(d)-2-horizon

    class CombinedStrategy(Strategy):
        def init(self):
            self.entry_bar=None
            self.decisions=[]

        def next(self):
            i=len(self.data)-1
            if self.position:
                if self.entry_bar is not None and i-self.entry_bar>=horizon-1:
                    self.position.close()
                return
            self.entry_bar=None
            if not bundle['enabled'] or self.data.index[-1]<test_start or i>last_entry or not np.isfinite(self.data.probability[-1]):
                return
            decision=plan(d.iloc[:i+1],float(self.data.probability[-1]),bundle['threshold'],float(self.equity),horizon)
            self.decisions.append(decision.to_dict())
            if decision.action=='WAIT': return
            # Next-open market execution can gap beyond the planned risk budget.
            qty=decision.quantity
            order=self.buy if decision.action=='BUY' else self.sell
            order(size=qty,sl=decision.stop,tp=decision.target)
            self.entry_bar=i+1

    renamed=d.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    bt=Backtest(renamed,CombinedStrategy,cash=10000,commission=.001,
                exclusive_orders=True,finalize_trades=True)
    stats=bt.run()
    curve=stats['_equity_curve'].loc[test_start:]
    summary={'period_start':str(test_start),'period_end':str(d.index[-1]),
             'trades':int(stats['# Trades']),
             'return_pct':float((curve.Equity.iloc[-1]/10000-1)*100),
             'max_drawdown_pct':float(((curve.Equity/curve.Equity.cummax())-1).min()*100),
             'win_rate_pct':float(stats['Win Rate [%]']) if stats['# Trades'] else None,
             'trading_enabled':bundle['enabled'],
             'decision_counts':pd.Series([x['reason'] for x in stats['_strategy'].decisions],dtype=str).value_counts().to_dict(),
             'execution':'Next daily open; 0.1% commission per side; intrabar ambiguity handled by backtesting.py; no borrow/funding fees.'}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'backtest.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    stats['_trades'].to_csv(output/'trades.csv',index=False)
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'NVDA_historical_features.csv')
    p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/strategy.joblib')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts')
    a=p.parse_args()
    run(a.data,a.bundle,a.output)
