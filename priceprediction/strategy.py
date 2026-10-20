"""Causal daily-bar signals and risk-constrained five-session order plans."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import norm
from .quant import causal_features, tail_risk, fractional_kelly

FEATURES = ['sma10', 'sma50', 'sma200', 'slope50', 'momentum20', 'rsi',
            'atr_pct', 'volatility', 'volume_z', 'adx', 'di_balance',
            'candle', 'support_distance', 'resistance_distance', 'fvg']


def prepare(bars):
    """All features at t use only candles <= t; levels exclude candle t."""
    d = bars.rename(columns=str.lower).copy()
    if not d.index.is_monotonic_increasing or d.index.has_duplicates:
        raise ValueError('Bars must have a unique, increasing timestamp index')
    o, h, l, c = (d[k].astype(float) for k in ('open', 'high', 'low', 'close'))
    if not np.isfinite(d[['open', 'high', 'low', 'close', 'volume']]).all().all():
        raise ValueError('OHLCV must be finite')
    if ((l <= 0) | (h < l) | (h < np.maximum(o, c)) | (l > np.minimum(o, c))).any():
        raise ValueError('Invalid OHLC candle')
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    for n in (10, 50, 200):
        d[f'ma{n}'] = c.rolling(n).mean()
        d[f'sma{n}'] = c/d[f'ma{n}']-1
    d['slope50'] = d.ma50.pct_change(10)
    d['momentum20'] = c.pct_change(20)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d['rsi'] = (gain/(gain+loss)).fillna(.5)
    d['atr_pct'] = d.atr/c
    d['log_return'] = np.log(c).diff()
    d['volatility'] = d.log_return.rolling(60).std()*np.sqrt(252)
    d['volume_z'] = (d.volume-d.volume.rolling(20).mean())/d.volume.rolling(20).std().replace(0, np.nan)
    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/14, adjust=False).mean()/d.atr
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/14, adjust=False).mean()/d.atr
    d['di_balance'] = ((plus-minus)/(plus+minus).replace(0, np.nan)).fillna(0)
    d['adx'] = d.di_balance.abs().ewm(alpha=1/14, adjust=False, min_periods=14).mean()*100
    body = (c-o).abs()
    span = (h-l).replace(0, np.nan)
    engulf_up = (c>o) & (c.shift()<o.shift()) & (o<=c.shift()) & (c>=o.shift())
    engulf_down = (c<o) & (c.shift()>o.shift()) & (o>=c.shift()) & (c<=o.shift())
    hammer = ((np.minimum(o,c)-l)>2*body) & ((h-np.maximum(o,c))<body) & (body/span>.1)
    star = ((h-np.maximum(o,c))>2*body) & ((np.minimum(o,c)-l)<body) & (body/span>.1)
    d['candle'] = engulf_up.astype(int)-engulf_down.astype(int)+hammer.astype(int)-star.astype(int)
    d['support'] = l.shift().rolling(20).min()
    d['resistance'] = h.shift().rolling(20).max()
    d['support_distance'] = (c-d.support)/d.atr
    d['resistance_distance'] = (d.resistance-c)/d.atr
    # Three-candle wick gaps, retained until fully filled or 20 bars old.
    zones, values = [], []
    for i in range(len(d)):
        zones = [(s,lo,hi,t) for s,lo,hi,t in zones
                 if i-t<=20 and not (s==1 and l.iloc[i]<=lo) and not (s==-1 and h.iloc[i]>=hi)]
        if i>=2:
            if l.iloc[i]>h.iloc[i-2]: zones.append((1,h.iloc[i-2],l.iloc[i],i))
            if h.iloc[i]<l.iloc[i-2]: zones.append((-1,h.iloc[i],l.iloc[i-2],i))
        a = d.atr.iloc[i]
        nearby = [s for s,lo,hi,t in zones if np.isfinite(a) and lo-a<=c.iloc[i]<=hi+a]
        values.append(int(np.sign(sum(nearby))))
    d['fvg'] = values
    bull = (c>d.ma200) & (d.ma50>d.ma200) & (d.slope50>0) & (d.di_balance>0) & (d.adx>=20)
    bear = (c<d.ma200) & (d.ma50<d.ma200) & (d.slope50<0) & (d.di_balance<0) & (d.adx>=20)
    d['regime'] = np.select([bull,bear],[1,-1],default=0)
    d=d.join(causal_features(c))
    return d


def black_scholes(spot, strike, years, volatility, rate=0.0, dividend=0.0):
    """European prices and risk-neutral P(S_T > strike); not real-world odds."""
    if not all(np.isfinite([spot,strike,years,volatility,rate,dividend])) or min(spot,strike,years,volatility)<=0:
        raise ValueError('Positive spot, strike, maturity and volatility required')
    d1=(np.log(spot/strike)+(rate-dividend+volatility**2/2)*years)/(volatility*np.sqrt(years))
    d2=d1-volatility*np.sqrt(years)
    return {'call':float(spot*np.exp(-dividend*years)*norm.cdf(d1)-strike*np.exp(-rate*years)*norm.cdf(d2)),
            'put':float(strike*np.exp(-rate*years)*norm.cdf(-d2)-spot*np.exp(-dividend*years)*norm.cdf(-d1)),
            'risk_neutral_above_strike':float(norm.cdf(d2))}


def simulate(returns, spot, stop, target, side, horizon=5, paths=4000, seed=42, cost=.002, block_size=1):
    """Bootstrap daily close paths; stop-first on a sampled barrier crossing.

    Daily paths miss intraday crossings. These are diagnostics, not calibrated odds.
    """
    r=np.asarray(returns,dtype=float)
    r=r[np.isfinite(r)][-252:]
    if len(r)<60 or horizon<1 or paths<1 or not 1<=block_size<=len(r): raise ValueError('Insufficient simulation history or invalid dimensions')
    if side not in (-1,1) or not np.isfinite([spot,stop,target,cost]).all() or min(spot,stop,target)<=0 or cost<0:
        raise ValueError('Invalid simulation prices, side or cost')
    if not (stop<spot<target if side==1 else target<spot<stop): raise ValueError('Invalid simulation bracket')
    rng=np.random.default_rng(seed)
    # Shrink noisy historical drift toward zero, preserving empirical shocks.
    centered=r-r.mean()+.25*r.mean()
    starts=rng.integers(0,len(r)-block_size+1,size=(paths,int(np.ceil(horizon/block_size))))
    indices=(starts[:,:,None]+np.arange(block_size)).reshape(paths,-1)[:,:horizon]
    draws=centered[indices]
    prices=spot*np.exp(np.cumsum(draws,axis=1))
    exit_prices=prices[:,-1].copy()
    active=np.ones(paths,dtype=bool)
    for j in range(horizon):
        stop_hit=active & ((prices[:,j]<=stop) if side==1 else (prices[:,j]>=stop))
        target_hit=active & ~stop_hit & ((prices[:,j]>=target) if side==1 else (prices[:,j]<=target))
        # Adverse gaps fill at sampled price, favorable crossings capped at target.
        exit_prices[stop_hit]=prices[stop_hit,j]
        exit_prices[target_hit]=target
        active &= ~(stop_hit|target_hit)
    net=side*(exit_prices/spot-1)-cost
    return {'profitable_probability':float((net>0).mean()), 'expected_net_return':float(net.mean()),
            'terminal_p05':float(np.quantile(prices[:,-1],.05)), 'terminal_p95':float(np.quantile(prices[:,-1],.95)),
            'var_95':tail_risk(net)['var'], 'expected_shortfall_95':tail_risk(net)['expected_shortfall'],
            'block_size':int(block_size)}


@dataclass
class Decision:
    action: str
    reason: str
    probability_up: float
    regime: str
    stop: float | None = None
    target: float | None = None
    quantity: int = 0
    monte_carlo: dict | None = None
    black_scholes: dict | None = None

    def to_dict(self): return asdict(self)


def plan(history, probability, threshold=.6, equity=10000., horizon=5, risk_fraction=.005, quant_risk=False):
    if not 0<=probability<=1 or not .5<=threshold<=1 or equity<=0 or not 0<risk_fraction<=.02:
        raise ValueError('Invalid probability, threshold, equity or risk budget')
    row=history.iloc[-1]
    regime={1:'bullish',-1:'bearish',0:'neutral'}[int(row.regime)]
    def wait(reason): return Decision('WAIT',reason,float(probability),regime)
    if not np.isfinite(row[FEATURES].astype(float)).all(): return wait('Indicator warmup incomplete')
    side=1 if probability>=threshold else -1 if probability<=1-threshold else 0
    if not side or side!=row.regime: return wait('Confidence or regime filter')
    price,atr=float(row.close),float(row.atr)
    near_support=0<=row.support_distance<=1.5
    near_resistance=0<=row.resistance_distance<=1.5
    confluence=(side*row.candle>0) or (side*row.fvg>0) or (near_support if side==1 else near_resistance)
    if not confluence: return wait('No price-action confirmation')
    risk=1.5*atr
    stop=price-side*risk
    target=price+side*2*risk
    # Avoid buying directly into resistance or selling into support.
    obstacle=float(row.resistance if side==1 else row.support)
    room=side*(obstacle-price)
    if 0<room<2*risk: return wait('Insufficient room to support/resistance')
    stop,target=round(stop,2),round(target,2)
    risk=abs(price-stop)
    if min(stop,target)<=0 or risk<=0 or side*(target-price)<=0 or side*(price-stop)<=0:
        return wait('Invalid bracket')
    mc=simulate(history.log_return,price,stop,target,side,horizon=horizon,block_size=min(5,horizon) if quant_risk else 1)
    bs=black_scholes(price,target,horizon/252,float(row.volatility))
    quantity=int(min(equity*risk_fraction/risk,equity*.2/price))
    if quant_risk:
        # These tighten exposure only; they never enable a rejected direction.
        # MC probabilities are for barrier outcomes, not model direction labels.
        kelly=fractional_kelly(mc['profitable_probability'],abs(target-price)/risk)
        es=mc['expected_shortfall_95']
        quantity=min(quantity,int(equity*kelly/price))
        if es>0: quantity=min(quantity,int(equity*risk_fraction/(price*es)))
        mc['fractional_kelly_notional_cap']=kelly
    if mc['profitable_probability']<.52 or mc['expected_net_return']<=0 or quantity<1:
        return Decision('WAIT','Simulation or position-size filter',float(probability),regime,monte_carlo=mc,black_scholes=bs)
    return Decision('BUY' if side==1 else 'SELL','Confidence, regime, structure and simulation agree',float(probability),regime,
                    stop,target,quantity,mc,bs)
