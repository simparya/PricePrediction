"""Numerical reference values, constraints and causality for public quant methods."""
import unittest
import numpy as np
import pandas as pd
from priceprediction.quant import (causal_features, kalman_level, fit_garch,
    garch_variance, tail_risk, fractional_kelly, hac_mean_interval,
    block_bootstrap_mean_interval, mean_variance_weights, inventory_quotes,
    liquidation_schedule, expected_credit_loss)
from priceprediction.strategy import simulate, prepare, plan
from unittest.mock import patch


class QuantTests(unittest.TestCase):
    def test_kalman_hand_computed_update(self):
        # P-=2, K=2/3 with initial covariance=Q=R=1.
        np.testing.assert_allclose(kalman_level([0,3],1,1),[0,2])
        np.testing.assert_allclose(kalman_level([4,4,4]),[4,4,4])

    def test_all_quant_features_are_prefix_invariant(self):
        rng=np.random.default_rng(42)
        close=pd.Series(100*np.exp(np.cumsum(rng.normal(0,.01,300))))
        pd.testing.assert_frame_equal(causal_features(close).iloc[:230],causal_features(close.iloc[:230]))
        self.assertTrue(np.isfinite(causal_features(close).iloc[100:]).all().all())

    def test_ewma_known_constant_log_return(self):
        close=pd.Series(np.exp(np.arange(100)*.01))
        self.assertAlmostEqual(causal_features(close).q_ewma_vol.iloc[-1],.01*np.sqrt(252),places=10)

    def test_garch_recursion_and_no_future_effect(self):
        params={'omega':.01,'alpha':.1,'beta':.8,'mean':0,'initial_variance':.1}
        h,next_h=garch_variance([.2,-.3],params)
        np.testing.assert_allclose(h,[.1,.094])
        self.assertAlmostEqual(next_h,.0942)
        short,_=garch_variance([.2],params)
        self.assertEqual(h[0],short[0])
        with self.assertRaises(ValueError): garch_variance([.1],{**params,'beta':.95})

    def test_garch_fit_is_stationary_and_finite(self):
        rng=np.random.default_rng(32)
        params=fit_garch(rng.normal(0,.015,400))
        self.assertGreater(params['omega'],0)
        self.assertLess(params['alpha']+params['beta'],1)
        h,next_h=garch_variance([.01,.02],params)
        self.assertTrue(np.isfinite(h).all())
        self.assertGreater(next_h,0)
        with self.assertRaises(ValueError): fit_garch(np.zeros(200))

    def test_expected_shortfall_fractional_tail(self):
        risk=tail_risk([-.1,-.2,-.3,-.4],confidence=.625)
        # Worst 1.5 observations: (0.4 + 0.5 * 0.3) / 1.5.
        self.assertAlmostEqual(risk['expected_shortfall'],.55/1.5)
        self.assertGreaterEqual(risk['expected_shortfall'],risk['var'])
        self.assertEqual(tail_risk([.01,.02])['expected_shortfall'],0)

    def test_kelly_no_edge_and_known_payoff(self):
        self.assertAlmostEqual(fractional_kelly(.6,2),.1)
        self.assertEqual(fractional_kelly(.1,2),0)
        self.assertEqual(fractional_kelly(1,2),.2)
        with self.assertRaises(ValueError): fractional_kelly(1.1,2)

    def test_hac_captures_serial_dependence(self):
        rng=np.random.default_rng(9)
        x=np.zeros(500)
        for i,e in enumerate(rng.normal(size=499),start=1): x[i]=.8*x[i-1]+e
        self.assertGreater(hac_mean_interval(x,5)['standard_error'],hac_mean_interval(x,0)['standard_error'])
        self.assertEqual(hac_mean_interval(np.ones(20),2)['lower'],1)

    def test_block_interval_deterministic_and_translation_invariant(self):
        x=np.arange(100)/100
        a=block_bootstrap_mean_interval(x)
        self.assertEqual(a,block_bootstrap_mean_interval(x))
        b=block_bootstrap_mean_interval(x+3)
        self.assertAlmostEqual(b['lower']-a['lower'],3)
        self.assertLess(a['lower'],a['mean'])
        self.assertGreater(a['upper'],a['mean'])

    def test_portfolio_solution_satisfies_first_order_condition(self):
        mu=np.array([.05,.08]);cov=np.array([[.04,.01],[.01,.09]])
        w=mean_variance_weights(mu,cov,3)
        np.testing.assert_allclose(3*cov@w,mu)
        with self.assertRaises(ValueError): mean_variance_weights(mu,np.ones((2,2)))

    def test_inventory_skew_and_liquidation_constraints(self):
        neutral=inventory_quotes(100,0,.2,1,.1,1)
        long=inventory_quotes(100,10,.2,1,.1,1)
        self.assertLess(neutral['bid'],neutral['ask'])
        self.assertLess(long['reservation_price'],neutral['reservation_price'])
        uniform=liquidation_schedule(100,5,risk_aversion=0)
        np.testing.assert_allclose(uniform['trade_sizes'],np.full(5,20))
        urgent=liquidation_schedule(100,5,risk_aversion=100)
        self.assertAlmostEqual(sum(urgent['trade_sizes']),100)
        self.assertGreater(urgent['trade_sizes'][0],urgent['trade_sizes'][-1])
        self.assertEqual(urgent['remaining'][-1],0)

    def test_research_build_cannot_replace_active_paper_model(self):
        from priceprediction.model_builder import build
        from priceprediction.paths import ARTIFACTS_DIR
        with self.assertRaisesRegex(ValueError,'overwrite'):
            build('unused.csv',ARTIFACTS_DIR/'strategy.joblib',quant=True)

    def test_credit_expected_loss(self):
        self.assertAlmostEqual(expected_credit_loss(.02,.4,100000),800)
        with self.assertRaises(ValueError): expected_credit_loss(-.1,.5,100)

    def test_block_simulation_reports_tail_risk(self):
        r=np.random.default_rng(4).normal(0,.02,252)
        a=simulate(r,100,95,110,1,block_size=5)
        self.assertEqual(a,simulate(r,100,95,110,1,block_size=5))
        self.assertGreaterEqual(a['expected_shortfall_95'],a['var_95'])
        self.assertEqual(a['block_size'],5)

    def test_quant_risk_only_reduces_position_size(self):
        rng=np.random.default_rng(7)
        close=100*np.exp(np.cumsum(rng.normal(.001,.01,300)))
        bars=pd.DataFrame({'open':close,'close':close,'high':close+1,'low':close-1,'volume':rng.integers(100,10000,300)})
        d=prepare(bars)
        idx=d.index[-1]
        d.loc[idx,['regime','candle','atr','resistance']]=[1,1,2,d.close.iloc[-1]-1]
        mc={'profitable_probability':.7,'expected_net_return':.01,'expected_shortfall_95':.2}
        with patch('priceprediction.strategy.simulate',return_value=mc.copy()):
            original=plan(d,.8,.6)
        with patch('priceprediction.strategy.simulate',return_value=mc.copy()):
            controlled=plan(d,.8,.6,quant_risk=True)
        self.assertEqual(original.action,'BUY')
        self.assertLessEqual(controlled.quantity,original.quantity)
        self.assertLessEqual(controlled.quantity*d.close.iloc[-1]*.2,50)


if __name__=='__main__': unittest.main()
