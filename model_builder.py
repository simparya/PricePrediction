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
from strategy import FEATURES, prepare

ROOT=Path(__file__).resolve().parent


def probability(bundle, frame):
    raw=np.clip(bundle['model'].predict_proba(frame[FEATURES])[:,1],1e-6,1-1e-6)
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


def build(data, output, horizon=5):
    if horizon<1: raise ValueError('Horizon must be positive')
    raw=pd.read_csv(data,index_col=0,parse_dates=True).sort_index()
    full=prepare(raw)
    # Future prices are used ONLY as training/evaluation labels.
    future=full.close.shift(-horizon)
    full['label']=(future>full.close).astype(int)
    full.loc[future.isna(),'label']=np.nan
    d=full.dropna(subset=FEATURES+['label']).copy()
    n=len(d)
    a,b,c=int(n*.5),int(n*.7),int(n*.85)
    if min(a-horizon,b-a-horizon,c-b-horizon,n-c)<40:
        raise ValueError('Need more data for purged train/calibration/validation/test splits')
    train,cal,val,test=d.iloc[:a-horizon],d.iloc[a:b-horizon],d.iloc[b:c-horizon],d.iloc[c:]
    model=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.03,
                        subsample=.8,colsample_bytree=.8,reg_lambda=5,
                        random_state=42,n_jobs=2)
    model.fit(train[FEATURES],train.label)
    p=np.clip(model.predict_proba(cal[FEATURES])[:,1],1e-6,1-1e-6)
    calibrator=LogisticRegression(C=1,random_state=42).fit(np.log(p/(1-p)).reshape(-1,1),cal.label)
    bundle={'model':model,'calibrator':calibrator,'features':FEATURES,'horizon':horizon,
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
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output)
    output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return bundle,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'NVDA_historical_features.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/strategy.joblib')
    parser.add_argument('--horizon',type=int,default=5)
    args=parser.parse_args()
    build(args.data,args.output,args.horizon)
