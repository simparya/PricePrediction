"""Offline, purged chronological training; never modifies the supplied model/data."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier
from .strategy import FEATURES, prepare
from .quant import QUANT_FEATURES, fit_garch, garch_variance, hac_mean_interval, block_bootstrap_mean_interval

from .paths import DATA_DIR, ARTIFACTS_DIR


def probability(bundle, frame):
    features=bundle.get('features',FEATURES)
    raw=np.clip(bundle['model'].predict_proba(frame[features])[:,1],1e-6,1-1e-6)
    return bundle['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]


def threshold_search(y, p, regime, minimum=25):
    """Validation directional precision with a Wilson lower confidence bound.

    Threshold selection is exploratory; the untouched test period is authoritative.
    """
    candidates=[]
    for t in (.50,.55,.60,.65,.70,.75,.80):
        take=((p>=t)&(regime==1))|((p<=1-t)&(regime==-1))
        n=int(take.sum())
        if n<minimum: continue
        acc=float(((p[take]>=.5)==y[take]).mean())
        z=1.96
        lower=(acc+z*z/(2*n)-z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n)))/(1+z*z/n)
        candidates.append({'threshold':t,'samples':n,'precision':acc,'wilson_lower':float(lower)})
    eligible=[x for x in candidates if x['precision']>=.55 and x['wilson_lower']>.5]
    best=max(eligible,key=lambda x:(x['wilson_lower'],x['samples'])) if eligible else None
    return best,candidates


def build(data, output, horizon=5, quant=False):
    if horizon<1: raise ValueError('Horizon must be positive')
    if quant and Path(output).resolve()==(ARTIFACTS_DIR/'strategy.joblib').resolve():
        raise ValueError('Quant research cannot overwrite the active paper model')
    raw=pd.read_csv(data,index_col=0,parse_dates=True).sort_index()
    full=prepare(raw)
    # Future prices are used ONLY as training/evaluation labels.
    future=full.close.shift(-horizon)
    full['label']=(future>full.close).astype(int)
    full.loc[future.isna(),'label']=np.nan
    features=FEATURES+QUANT_FEATURES if quant else FEATURES
    d=full.dropna(subset=features+['label']).copy()
    n=len(d)
    a,b,c=int(n*.5),int(n*.7),int(n*.85)
    if min(a-horizon,b-a-horizon,c-b-horizon,n-c)<40:
        raise ValueError('Need more data for purged train/calibration/validation/test splits')
    train,cal,val,test=d.iloc[:a-horizon],d.iloc[a:b-horizon],d.iloc[b:c-horizon],d.iloc[c:]
    model=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.03,
                        subsample=.8,colsample_bytree=.8,reg_lambda=5,
                        random_state=42,n_jobs=2)
    model.fit(train[features],train.label)
    p=np.clip(model.predict_proba(cal[features])[:,1],1e-6,1-1e-6)
    calibrator=LogisticRegression(C=1,random_state=42).fit(np.log(p/(1-p)).reshape(-1,1),cal.label)
    bundle={'model':model,'calibrator':calibrator,'features':features,'horizon':horizon,'quant_enabled':bool(quant),
            'test_start':str(test.index[0]), 'threshold':1., 'enabled':False}
    pv=probability(bundle,val)
    best,candidates=threshold_search(val.label.to_numpy(),pv,val.regime.to_numpy())
    if best: bundle.update(threshold=best['threshold'],enabled=True)
    pt=probability(bundle,test)
    take=(((pt>=bundle['threshold'])&(test.regime==1))|((pt<=1-bundle['threshold'])&(test.regime==-1))) & bundle['enabled']
    # Price-only baseline on identical splits, without new structure/regime features.
    base_features=['sma10','sma50','sma200','rsi','atr_pct','volume_z']
    baseline=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.03,random_state=42,n_jobs=2)
    baseline.fit(train[base_features],train.label)
    bp=baseline.predict_proba(test[base_features])[:,1]
    report={'horizon_sessions':horizon,'rows':n,
            'splits':{name:{'rows':len(part),'start':str(part.index[0]),'end':str(part.index[-1])}
                      for name,part in [('train',train),('calibration',cal),('validation',val),('test',test)]},
            'purge_sessions_at_each_boundary':horizon,'threshold_candidates':candidates,
            'selected':best,'trading_enabled':bundle['enabled'],
            'test':{'accuracy':float(accuracy_score(test.label,pt>=.5)),
                    'brier':float(brier_score_loss(test.label,pt)),
                    'baseline_accuracy':float(accuracy_score(test.label,bp>=.5)),
                    'baseline_brier':float(brier_score_loss(test.label,bp)),
                    'majority_class_accuracy':float(max(test.label.mean(),1-test.label.mean())),
                    'qualified_directional_signals':int(take.sum()),
                    'qualified_directional_precision':float(((pt[take]>=.5)==test.label[take]).mean()) if take.any() else None},
            'limitations':['Overlapping five-day labels make signal observations dependent; Wilson bounds are heuristic.',
                          'Cached OHLCV adjustment and timestamp provenance are not verified.',
                          'No claim of improved future accuracy; test returns exclude funding/borrow fees.',
                          'Black-Scholes uses historical volatility, zero rate/dividend; no market option quotes.']}
    if quant:
        # These scores use every test timestamp; HAC/bootstrap retain temporal order.
        calibrated_errors=(pt-test.label.to_numpy())**2
        baseline_errors=(bp-test.label.to_numpy())**2
        improvement=baseline_errors-calibrated_errors
        report['quant_research']={'features':QUANT_FEATURES,
            'brier_improvement_vs_price_baseline_hac':hac_mean_interval(improvement,lags=horizon),
            'brier_improvement_vs_price_baseline_block_bootstrap':block_bootstrap_mean_interval(improvement,block=horizon),
            'test_accuracy_block_bootstrap':block_bootstrap_mean_interval(((pt>=.5)==test.label.to_numpy()).astype(float),block=horizon),
            'garch_fit_scope':'Training returns only; parameters frozen for later volatility updates',
            'active_paper_model_replaced':False,
            'test_reuse_warning':'This historical holdout has been inspected before. Results are exploratory; collect new forward data.'}
        try:
            parameters=fit_garch(train.log_return.to_numpy())
            bundle['garch_parameters']=parameters
            _,next_variance=garch_variance(d.log_return.to_numpy(),parameters)
            report['quant_research']['garch11']={'parameters':parameters,'next_daily_volatility':float(np.sqrt(next_variance)),
                'boundary_fit_warning':bool(parameters['alpha']<1e-6 or parameters['beta']>.997 or parameters['alpha']+parameters['beta']>.9985)}
        except ValueError as error:
            report['quant_research']['garch11']={'unavailable':str(error)}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output)
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return bundle,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA_DIR/'NVDA_historical_features.csv')
    parser.add_argument('--output',type=Path,help='Default: separate research bundle with --quant; active bundle otherwise')
    parser.add_argument('--quant',action='store_true',help='Train an isolated candidate using public quant features')
    parser.add_argument('--horizon',type=int,default=5)
    args=parser.parse_args()
    output=args.output or (ARTIFACTS_DIR/'research/strategy.joblib' if args.quant else ARTIFACTS_DIR/'strategy.joblib')
    if args.quant and output.resolve()==(ARTIFACTS_DIR/'strategy.joblib').resolve():
        parser.error('--quant cannot overwrite the active paper model; use a separate output')
    build(args.data,output,args.horizon,quant=args.quant)
