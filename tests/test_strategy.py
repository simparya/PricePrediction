import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from priceprediction.strategy import prepare, FEATURES, black_scholes, simulate, plan
from priceprediction.model_builder import threshold_search


def bars(n=300):
    rng=np.random.default_rng(123)
    c=100*np.exp(np.cumsum(rng.normal(.001,.01,n)))
    o=np.r_[c[0],c[:-1]]
    return pd.DataFrame({'open':o,'close':c,'high':np.maximum(o,c)+1,
                         'low':np.minimum(o,c)-1,'volume':rng.integers(1000,10000,n)},
                        index=pd.date_range('2020-01-01',periods=n,tz='UTC'))


class StrategyTests(unittest.TestCase):
    def test_future_candles_cannot_change_past_features(self):
        d=bars()
        prefix=prepare(d.iloc[:250])
        full=prepare(d)
        pd.testing.assert_frame_equal(prefix[FEATURES+['regime','support','resistance']],
                                      full.iloc[:250][FEATURES+['regime','support','resistance']])
        changed=d.copy()
        changed.iloc[250:,changed.columns.get_indexer(['open','close','high','low'])]*=10
        pd.testing.assert_frame_equal(full.iloc[:250],prepare(changed).iloc[:250])

    def test_support_resistance_exclude_current_candle(self):
        d=bars()
        d.iloc[-1,d.columns.get_loc('high')]=10000
        f=prepare(d)
        self.assertEqual(f.resistance.iloc[-1],d.high.iloc[-21:-1].max())

    def test_black_scholes_known_price_and_parity(self):
        v=black_scholes(100,100,1,.2,.05)
        self.assertAlmostEqual(v['call'],10.45058357,places=6)
        self.assertAlmostEqual(v['call']-v['put'],100-100*np.exp(-.05),places=8)
        with self.assertRaises(ValueError): black_scholes(100,100,0,.2)

    def test_simulation_barrier_and_determinism(self):
        r=np.full(252,-.08)
        v=simulate(r,100,95,110,1)
        self.assertEqual(v,simulate(r,100,95,110,1))
        self.assertEqual(v['profitable_probability'],0)
        self.assertLess(v['expected_net_return'],0)
        self.assertEqual(simulate(r,100,105,90,-1)['profitable_probability'],1)

    def test_bracket_geometry_and_risk_budget(self):
        d=prepare(bars())
        idx=d.index[-1]
        d.loc[idx,['regime','candle','atr','resistance','volatility']]=[1,1,2,d.close.iloc[-1]-1,.3]
        with patch('priceprediction.strategy.simulate',return_value={'profitable_probability':.7,'expected_net_return':.02}):
            decision=plan(d,.8,.6,10000)
        self.assertEqual(decision.action,'BUY')
        self.assertLess(decision.stop,d.close.iloc[-1])
        self.assertGreater(decision.target,d.close.iloc[-1])
        self.assertLessEqual(decision.quantity*abs(d.close.iloc[-1]-decision.stop),50)
        self.assertLessEqual(decision.quantity*d.close.iloc[-1],2000)
        self.assertEqual(plan(d,.51,.6).action,'WAIT')
        d.loc[idx,'regime']=-1
        self.assertEqual(plan(d,.8,.6).action,'WAIT')

    def test_no_threshold_when_no_evidence(self):
        best,_=threshold_search(np.tile([0,1],50),np.full(100,.6),np.ones(100))
        self.assertIsNone(best)

    def test_fvg_expires_when_fully_filled(self):
        d=bars()
        d.loc[:,['open','close','high','low']]=[100.,100.,101.,99.]
        d.iloc[-3:,d.columns.get_indexer(['open','close','high','low'])]=[[104,104,105,103],[104,104,105,103],[102,102,104,100]]
        f=prepare(d)
        self.assertEqual(f.fvg.iloc[-2],1)
        self.assertEqual(f.fvg.iloc[-1],0)

    def test_backtest_enters_next_open_and_exits_after_five_sessions(self):
        import tempfile
        from pathlib import Path
        import joblib
        from priceprediction.strategy import Decision
        from priceprediction import backtester
        d=bars()
        start=d.index[240]
        calls=[]
        def decision(history,*args,**kwargs):
            calls.append(len(history))
            if len(calls)==1:
                return Decision('BUY','test',.8,'bullish',1.,10000.,1)
            return Decision('WAIT','test',.8,'bullish')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            d.to_csv(root/'data.csv')
            joblib.dump({'test_start':str(start),'horizon':5,'enabled':True,'threshold':.6},root/'bundle.joblib')
            with patch('priceprediction.backtester.probability',side_effect=lambda b,f:np.full(len(f),.8)), patch('priceprediction.backtester.plan',side_effect=decision):
                report=backtester.run(root/'data.csv',root/'bundle.joblib',root/'out')
            trades=pd.read_csv(root/'out/trades.csv')
            self.assertEqual(report['trades'],1)
            self.assertEqual(trades.EntryBar.iloc[0],241)
            self.assertEqual(trades.ExitBar.iloc[0]-trades.EntryBar.iloc[0],5)
            self.assertAlmostEqual(trades.EntryPrice.iloc[0],d.open.iloc[241])

    def test_no_import_time_trading_or_training(self):
        from priceprediction import main, backtester, execution
        self.assertTrue(callable(main.propose))


if __name__=='__main__': unittest.main()
