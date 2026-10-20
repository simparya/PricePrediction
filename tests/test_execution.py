import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import pandas as pd
from alpaca.trading.enums import OrderSide
from priceprediction.execution import execute


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.client=Mock()
        self.client.get_clock.return_value=NS(is_open=True,timestamp=pd.Timestamp('2026-07-14 15:00Z'))
        self.client.get_orders.return_value=[]
        self.client.get_all_positions.return_value=[]
        self.client.get_calendar.return_value=[1,2]
        self.client.get_account.return_value=NS(equity='10000',buying_power='10000')
        self.decision={'action':'BUY','signal_bar':'2026-07-13 04:00Z','stop':95.,'target':110.,'quantity':10}

    def test_paper_only_and_bracket_geometry(self):
        with patch('priceprediction.execution.TradingClient',return_value=self.client) as ctor:
            execute(self.decision,'test-key','test-secret',100.)
        ctor.assert_called_once_with('test-key','test-secret',paper=True)
        order=self.client.submit_order.call_args.kwargs['order_data']
        self.assertEqual(order.qty,10)
        self.assertLess(order.stop_loss.stop_price,order.limit_price)
        self.assertGreater(order.take_profit.limit_price,order.limit_price)

    def test_submitted_entry_is_audited_with_broker_id(self):
        import tempfile
        import sqlite3
        from pathlib import Path
        self.client.submit_order.return_value=NS(id='paper-order-123')
        decision={**self.decision,'probability_up':.8}
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'audit.db'
            with patch('priceprediction.execution.TradingClient',return_value=self.client):
                execute(decision,'k','s',100.,audit_path=target)
            with sqlite3.connect(target) as db:
                record=db.execute('SELECT qty,alpaca_order_id,status FROM trades').fetchone()
            self.assertEqual(record,(10,'paper-order-123','SUBMITTED'))

    def test_duplicate_signal_is_not_resubmitted(self):
        self.client.get_orders.return_value=[NS(client_order_id='combined-NVDA-20260713-buy',filled_at=None)]
        with patch('priceprediction.execution.TradingClient',return_value=self.client):
            self.assertIn('already processed',execute(self.decision,'k','s',100.))
        self.client.submit_order.assert_not_called()

    def test_unmanaged_position_is_preserved(self):
        self.client.get_all_positions.return_value=[NS(symbol='NVDA')]
        with patch('priceprediction.execution.TradingClient',return_value=self.client):
            with self.assertRaisesRegex(RuntimeError,'Unmanaged'):
                execute(self.decision,'k','s',100.)
        self.client.close_position.assert_not_called()

    def test_five_session_exit_without_new_entry(self):
        filled=pd.Timestamp('2026-07-06 15:00Z')
        order=NS(id='id',client_order_id='combined-NVDA-20260702-buy',filled_at=filled,
                 side=OrderSide.BUY,filled_qty='10')
        self.client.get_orders.side_effect=[[order],[],[]]
        self.client.get_all_positions.return_value=[NS(symbol='NVDA',side=NS(value='long'),qty='10')]
        self.client.get_calendar.return_value=list(range(6))
        with patch('priceprediction.execution.TradingClient',return_value=self.client):
            self.assertIn('exit submitted',execute({'action':'WAIT'},'k','s',100.))
        self.client.close_position.assert_called_once_with('NVDA')
        self.client.submit_order.assert_not_called()


if __name__=='__main__': unittest.main()
