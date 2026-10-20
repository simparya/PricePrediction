"""Public quantitative methods. See docs/QUANT_METHODS.md for equations/sources.

No proprietary firm models are reproduced. Portfolio/execution/credit utilities
are standalone research calculations, never broker instructions.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

QUANT_FEATURES = ['q_ewma_vol', 'q_kalman_gap', 'q_ou_z', 'q_ou_strength',
                  'q_return_autocorr', 'q_downside_vol']


def _finite(values, minimum=1):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<minimum or not np.isfinite(x).all():
        raise ValueError('Need a finite one-dimensional sample of sufficient size')
    return x


def kalman_level(values, process_variance=1e-5, observation_variance=1e-3):
    x=_finite(values)
    if process_variance<=0 or observation_variance<=0:
        raise ValueError('State and observation variances must be positive')
    level=x[0]
    covariance=observation_variance
    filtered=np.empty(len(x))
    filtered[0]=level
    for i in range(1,len(x)):
        predicted_cov=covariance+process_variance
        gain=predicted_cov/(predicted_cov+observation_variance)
        level+=gain*(x[i]-level)
        covariance=(1-gain)*predicted_cov
        filtered[i]=level
    return filtered


def causal_features(close, decay=.94, window=60):
    if not 0<decay<1 or window<5: raise ValueError('Invalid decay or lookback')
    c=pd.Series(close,copy=True).astype(float)
    if not np.isfinite(c).all() or (c<=0).any(): raise ValueError('Prices must be finite and positive')
    log=np.log(c)
    r=log.diff()
    out=pd.DataFrame(index=c.index)
    # Includes r_t: estimate of next-session volatility known at close t.
    variance=r.pow(2).ewm(alpha=1-decay,adjust=False,min_periods=window).mean()
    out['q_ewma_vol']=np.sqrt(252*variance)
    level=pd.Series(kalman_level(log),index=c.index)
    out['q_kalman_gap']=(log-level)/np.sqrt(variance).replace(0,np.nan)
    ou_z=np.full(len(c),np.nan)
    strength=np.full(len(c),np.nan)
    for i in range(window,len(c)):
        segment=log.iloc[i-window:i+1].to_numpy()
        x,y=segment[:-1],segment[1:]
        centered=x-x.mean()
        denom=centered@centered
        phi=float(centered@(y-y.mean())/denom) if denom>1e-14 else 1.
        # A stationary AR(1) with positive phi admits an OU interpretation.
        if 0<phi<1:
            intercept=y.mean()-phi*x.mean()
            equilibrium=intercept/(1-phi)
            residual=y-intercept-phi*x
            stationary_std=np.sqrt(np.mean(residual**2)/(1-phi**2))
            ou_z[i]=(log.iloc[i]-equilibrium)/max(stationary_std,1e-10)
            strength[i]=-np.log(phi)
        else:
            ou_z[i]=0.
            strength[i]=0.
    out['q_ou_z']=ou_z
    out['q_ou_strength']=strength
    out['q_return_autocorr']=r.rolling(window).corr(r.shift()).fillna(0)
    out['q_downside_vol']=np.sqrt(r.clip(upper=0).pow(2).rolling(window).mean()*252)
    return out


def garch_variance(returns, parameters):
    """Predictive daily variances h_t and h_(n+1), with frozen fitted parameters."""
    r=_finite(returns)
    omega,alpha,beta=(float(parameters[k]) for k in ('omega','alpha','beta'))
    if omega<=0 or min(alpha,beta)<0 or alpha+beta>=1:
        raise ValueError('Stationary GARCH parameters required')
    h=float(parameters.get('initial_variance',omega/(1-alpha-beta)))
    if not np.isfinite(h) or h<=0: raise ValueError('Invalid variance initialization')
    mean=float(parameters.get('mean',0.))
    predictions=[]
    for innovation in r-mean:
        predictions.append(h)
        h=omega+alpha*innovation**2+beta*h
    return np.asarray(predictions),float(h)


def fit_garch(returns):
    """Gaussian quasi-MLE; call ONLY on a training sample, never in prepare()."""
    r=_finite(returns,minimum=100)
    mean=float(r.mean())
    scaled=(r-mean)*100
    variance=float(np.var(scaled))
    if variance<=1e-12: raise ValueError('Cannot fit GARCH to constant returns')
    def objective(theta):
        omega,alpha,beta=theta
        h=variance
        value=0.
        for innovation in scaled:
            value+=np.log(h)+innovation**2/h
            h=omega+alpha*innovation**2+beta*h
        return .5*value/len(scaled)
    result=minimize(objective,[variance*.05,.05,.9],method='SLSQP',
                    bounds=[(1e-8,100*variance),(0,.998),(0,.998)],
                    constraints=[{'type':'ineq','fun':lambda t:.999-t[1]-t[2]}],
                    options={'maxiter':300,'ftol':1e-9})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('GARCH optimizer did not converge')
    return {'omega':float(result.x[0]/10000),'alpha':float(result.x[1]),
            'beta':float(result.x[2]),'mean':mean,'initial_variance':variance/10000}


def tail_risk(returns, confidence=.95):
    r=_finite(returns,minimum=2)
    if not 0<confidence<1: raise ValueError('Confidence must be in (0, 1)')
    # Empirical upper-tail loss average, including a fractional boundary weight.
    losses=np.sort(-r)[::-1]
    mass=(1-confidence)*len(losses)
    whole=int(np.floor(mass))
    fraction=mass-whole
    total=losses[:whole].sum()
    if fraction>0: total+=fraction*losses[min(whole,len(losses)-1)]
    return {'var':float(max(0,np.quantile(losses,confidence))),
            'expected_shortfall':float(max(0,total/mass))}


def fractional_kelly(win_probability, payoff_ratio, fraction=.25, cap=.2):
    if not 0<=win_probability<=1 or payoff_ratio<=0 or not 0<fraction<=1 or not 0<cap<=1:
        raise ValueError('Invalid Kelly inputs')
    full=win_probability-(1-win_probability)/payoff_ratio
    return float(np.clip(fraction*full,0,cap))


def hac_mean_interval(values, lags=5, confidence=.95):
    x=_finite(values,minimum=3)
    if not 0<=lags<len(x) or not 0<confidence<1: raise ValueError('Invalid HAC parameters')
    residual=x-x.mean()
    long_variance=float(residual@residual/len(x))
    for lag in range(1,lags+1):
        gamma=float(residual[lag:]@residual[:-lag]/len(x))
        long_variance+=2*(1-lag/(lags+1))*gamma
    se=np.sqrt(max(0,long_variance)/len(x))
    z=norm.ppf((1+confidence)/2)
    return {'mean':float(x.mean()),'standard_error':float(se),
            'lower':float(x.mean()-z*se),'upper':float(x.mean()+z*se),'lags':int(lags)}


def block_bootstrap_mean_interval(values, block=5, samples=2000, seed=42):
    x=_finite(values,minimum=10)
    if not 1<=block<=len(x) or samples<100: raise ValueError('Invalid bootstrap dimensions')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x)-block+1,size=(samples,int(np.ceil(len(x)/block))))
    indices=(starts[:,:,None]+np.arange(block)).reshape(samples,-1)[:,:len(x)]
    means=x[indices].mean(axis=1)
    return {'mean':float(x.mean()),'lower':float(np.quantile(means,.025)),
            'upper':float(np.quantile(means,.975)),'block':int(block),'samples':int(samples)}


def mean_variance_weights(expected_returns, covariance, risk_aversion=5.):
    """Unconstrained risky weights, with remaining wealth in cash; permits shorts."""
    mu=_finite(expected_returns)
    cov=np.asarray(covariance,dtype=float)
    if cov.shape!=(len(mu),len(mu)) or not np.isfinite(cov).all() or risk_aversion<=0:
        raise ValueError('Invalid covariance/risk aversion')
    if not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min()<=0:
        raise ValueError('Positive definite symmetric covariance required')
    return np.linalg.solve(cov,mu)/risk_aversion


def inventory_quotes(mid, inventory, volatility, time_remaining, risk_aversion, liquidity):
    """Avellaneda–Stoikov stylized quotes; absolute-price Brownian volatility."""
    if not np.isfinite([mid,inventory,volatility,time_remaining,risk_aversion,liquidity]).all():
        raise ValueError('Finite inputs required')
    if mid<=0 or min(volatility,time_remaining)<0 or min(risk_aversion,liquidity)<=0:
        raise ValueError('Invalid quote parameters')
    reservation=mid-inventory*risk_aversion*volatility**2*time_remaining
    spread=risk_aversion*volatility**2*time_remaining+2/risk_aversion*np.log1p(risk_aversion/liquidity)
    bid,ask=reservation-spread/2,reservation+spread/2
    if bid<=0: raise ValueError('Parameters imply a nonpositive bid')
    return {'bid':float(bid),'ask':float(ask),'reservation_price':float(reservation)}


def liquidation_schedule(shares, intervals=5, duration=1., volatility=.02, risk_aversion=1., impact=.1):
    """Continuous Almgren–Chriss linear-impact inventory sampled on a time grid."""
    if not np.isfinite([shares,duration,volatility,risk_aversion,impact]).all() or shares<0 or intervals<1 or duration<=0 or volatility<0 or risk_aversion<0 or impact<=0:
        raise ValueError('Invalid execution parameters')
    t=np.linspace(0,duration,intervals+1)
    k=np.sqrt(risk_aversion*volatility**2/impact)
    if k*duration<1e-7:
        remaining=shares*(1-t/duration)
    else:
        # Algebraically equivalent ratio avoids overflow for high urgency.
        remaining=shares*np.exp(-k*t)*(-np.expm1(-2*k*(duration-t)))/(-np.expm1(-2*k*duration))
    return {'remaining':remaining.tolist(),'trade_sizes':(-np.diff(remaining)).tolist()}


def expected_credit_loss(probability_default, loss_given_default, exposure):
    if not np.isfinite([probability_default,loss_given_default,exposure]).all() or not 0<=probability_default<=1 or not 0<=loss_given_default<=1 or exposure<0:
        raise ValueError('Invalid PD, LGD or EAD')
    return float(probability_default*loss_given_default*exposure)
