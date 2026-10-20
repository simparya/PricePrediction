"""Explicit paper-only order execution and five-session time exits.

Run once per open-market session. No unattended loop or real-money endpoint.
"""
import time
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (LimitOrderRequest, TakeProfitRequest, StopLossRequest,
                                     GetOrdersRequest, GetCalendarRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

PREFIX='combined-NVDA-'


def execute(decision,key,secret,signal_close,horizon=5):
    client=TradingClient(key,secret,paper=True)
    clock=client.get_clock()
    if not clock.is_open: raise RuntimeError('Market is closed; no action taken')
    now=pd.Timestamp(clock.timestamp)
    closed=client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED,symbols=['NVDA'],limit=500))
    entries=[o for o in closed if o.client_order_id.startswith(PREFIX) and o.filled_at is not None]
    position=next((p for p in client.get_all_positions() if p.symbol=='NVDA'),None)
    if position:
        if not entries: raise RuntimeError('Unmanaged NVDA position; refusing to change it')
        entry=max(entries,key=lambda o:o.filled_at)
        latest_fill=max((o for o in closed if o.filled_at is not None),key=lambda o:o.filled_at)
        if latest_fill.id!=entry.id or abs(float(position.qty))!=float(entry.filled_qty):
            raise RuntimeError('Cannot establish exclusive ownership of position; manual review required')
        if (entry.side==OrderSide.BUY)!=(str(position.side.value)=='long'):
            raise RuntimeError('Position direction does not match managed entry')
        start=pd.Timestamp(entry.filled_at).tz_convert('America/New_York').date()
        end=now.tz_convert('America/New_York').date()
        sessions=client.get_calendar(filters=GetCalendarRequest(start=start,end=end))
        if len(sessions)<=horizon: return 'HOLD: time exit not due'
        orders=client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN,symbols=['NVDA']))
        for order in orders: client.cancel_order_by_id(order.id)
        # Bounded cancellation wait; never submit liquidation against pending brackets.
        for _ in range(15):
            if not client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN,symbols=['NVDA'])):
                # A bracket may have filled while cancellation was in flight.
                if any(p.symbol=='NVDA' for p in client.get_all_positions()):
                    client.close_position('NVDA')
                    return 'Five-session paper exit submitted'
                return 'Position already closed'
            time.sleep(1)
        raise RuntimeError('Bracket cancellation not confirmed; time exit deferred')
    if decision['action']=='WAIT': return 'WAIT: no entry'
    signal=pd.Timestamp(decision['signal_bar'])
    sessions=client.get_calendar(filters=GetCalendarRequest(
        start=signal.tz_convert('America/New_York').date(),end=now.tz_convert('America/New_York').date()))
    if len(sessions)!=2: raise RuntimeError('Signal is not from the previous trading session')
    side=OrderSide.BUY if decision['action']=='BUY' else OrderSide.SELL
    order_id=f'{PREFIX}{signal.strftime("%Y%m%d")}-{side.value}'
    if any(o.client_order_id==order_id for o in closed): return 'Signal already processed'
    if client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN,symbols=['NVDA'])):
        return 'NVDA orders already pending'
    if side==OrderSide.SELL and not client.get_asset('NVDA').shortable:
        raise RuntimeError('NVDA is not currently shortable')
    account=client.get_account()
    limit=round(signal_close,2)
    stop,target=decision['stop'],decision['target']
    valid=stop<limit<target if side==OrderSide.BUY else target<limit<stop
    if not valid: raise ValueError('Invalid rounded bracket')
    qty=min(decision['quantity'],int(float(account.equity)*.005/abs(limit-stop)),
            int(float(account.buying_power)*.2/limit),int(float(account.equity)*.2/limit))
    if qty<1: raise RuntimeError('Insufficient buying power or risk budget')
    client.submit_order(order_data=LimitOrderRequest(
        symbol='NVDA',qty=qty,side=side,limit_price=limit,
        time_in_force=TimeInForce.DAY,order_class=OrderClass.BRACKET,
        client_order_id=order_id,take_profit=TakeProfitRequest(limit_price=target),
        stop_loss=StopLossRequest(stop_price=stop)))
    return 'Paper limit bracket submitted'
