"""Coverage for the integration of main's configuration and audit utilities."""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from priceprediction.config import get_alpaca_credentials
from priceprediction.main import audit_decision


class IntegrationTests(unittest.TestCase):
    def test_both_credential_naming_conventions(self):
        for names in [('ALPACA_API_KEY','ALPACA_SECRET_KEY'),('APCA_API_KEY_ID','APCA_API_SECRET_KEY')]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp:
                with patch('priceprediction.config.ROOT',Path(tmp)), patch.dict(os.environ,{names[0]:'test-key',names[1]:'test-secret'},clear=True):
                    self.assertEqual(get_alpaca_credentials(),('test-key','test-secret'))

    def test_env_file_and_missing_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('priceprediction.config.ROOT',root), patch.dict(os.environ,{},clear=True):
                with self.assertRaises(RuntimeError): get_alpaca_credentials()
                (root/'.env').write_text('APCA_API_KEY_ID=test-key\nAPCA_API_SECRET_KEY=test-secret\n')
                self.assertEqual(get_alpaca_credentials(),('test-key','test-secret'))

    def test_proposal_audit_uses_separate_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'artifacts/audit.db'
            audit_decision({'probability_up':.6,'price':100,'atr':2,'regime':'neutral','action':'WAIT'},target)
            with sqlite3.connect(target) as db:
                result=db.execute('SELECT symbol,prob_up,action FROM predictions').fetchone()
            self.assertEqual(result,('NVDA',.6,'WAIT'))


if __name__=='__main__': unittest.main()
