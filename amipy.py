#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:26:03 2017
@author: github.com/Quantmatic
"""
import time
from itertools import izip, count
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pymongo import MongoClient
import numpy as np
from ffn import PerformanceStats
import numba


def df_resample(dframe, interval):
    """ resample 1m data """
    _result = dframe.resample(interval).agg({'open': 'first',
                                             'high': 'max',
                                             'low': 'min',
                                             'close': 'last',
                                             'volume': 'sum',
                                             'oi': 'last'})
    _df = pd.DataFrame(_result).dropna(axis=0)

    _df[['volume', 'oi']] = _df[['volume', 'oi']].fillna(0.0).astype(int)
    return _df[:][['open', 'high', 'low', 'close', 'volume', 'oi']]


def mongo_grab(symbol, dbname, startdate, enddate, interval='60min', resample=False):
    ''' Grab the required strech of quotes from MongoDB '''
    client = MongoClient()
    dbase = client[dbname]
    collection = dbase[symbol]
    start = time.time()
    cursor = collection.find({'datetime': {'$gte': startdate, '$lt': enddate}})
    data = list(cursor)
    _df = pd.DataFrame(data)
    print 'Symbol: ' + symbol
    print 'Collection retrived in ' + str(time.time() - start) + \
        ' seconds. ' + str(len(_df))+' Bars.\n'
    _df.set_index('datetime', drop=False, append=False, inplace=True, verify_integrity=False)
    _df.index = pd.to_datetime(_df.index)
    _df = _df[:][['open', 'high', 'low', 'close', 'volume', 'oi']]
    if not resample:
        return _df
    else:
        start = time.time()
        resampled_df = df_resample(_df, interval)
        print 'Resample finished in ' + str(time.time() - start) + ' seconds.\n'
        return resampled_df


@numba.jit
def __remove(array1, array2, maxtrades=1):
    """ Remove excessive signals """
    nnn = len(array1)
    output = np.zeros(nnn, dtype='int64')
    i = 0

    while i < nnn:
        if array1[i]:
            output[i] = 1
            count = 1
            j = i+1
            while j < nnn:
                if array1[j]:
                    count += 1
                    if count > maxtrades:
                        output[j] = 0
                    else:
                        output[j] = 1

                if array2[j]:
                    break

                j += 1

            i = j
        else:
            i += 1

    return output


def ex_rem(array1, array2, maxtrades=1):
    """ Remove excessive signals """
    assert array1.index.equals(array2.index), 'Indices do not match'
    idx = array1.index
    ret = __remove(array1.values, array2.values, maxtrades)
    return pd.Series(ret, index=idx, dtype=bool)


def max_draw(trades):
    ''' calculate draw down '''
    maxeq = 0
    maxdd = 0 #maxdd
    closs = 0 #consecutive losses
    trades = trades.equity.sort_index()
    trades = trades[trades != trades.shift(1)].values
    iterator = -1

    for i, item in izip(count(), trades):
        if i > iterator:
            if item > maxeq:
                maxeq = item

            loss = 0
            drawd = 0
            cnt = i+1
            for j, col in enumerate(trades[cnt:]):
                if col < trades[j+cnt-1]:
                    loss += 1
                    drawd = (item - col) / item
                    if drawd > maxdd:
                        maxdd = drawd
                    if loss > closs:
                        closs = loss

                if col > maxeq:
                    iterator = j
                    break
                if col > trades[j+cnt-1]:
                    loss = 0

    return maxdd*100, closs

def _max_rolling_dd(ser):
    ''' max dd calculations '''
    max2here = pd.Series(ser).expanding().max()
    dd2here = ser - max2here
    return dd2here.min()

class Amipy(object):
    """ initialize constants required for backtest """
    def __init__(self, SYMBOL, EQUITY, MARGIN, TICKVALUE, TICKSIZE, RISK, DATA):
        self.symbol = SYMBOL
        self.starting_equity = EQUITY
        self.margin_required = MARGIN
        self.tickvalue = TICKVALUE
        self.tick_size = TICKSIZE
        self.risk = RISK
        self.trades = None
        self.imp_equity = None
        self.ohlc = DATA

    @numba.jit
    def apply_stops_cover(self, buy, short, shortprice, stoploss, takeprofit):
        ''' apply stops on short trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        shortprice = shortprice.values
        _open = self.ohlc.open.values
        mcover = np.zeros(len(short), dtype='int64')

        nnn = len(buy)
        for i in xrange(nnn):
            if short[i]:
                topen = shortprice[i]
                for cnt in xrange(i+1, nnn, 1):
                    val = topen - _open[cnt]
                    if val > takeprofit * tsize:
                        mcover[cnt] = 1
                        break
                    elif val < -stoploss * tsize:
                        mcover[cnt] = 1
                        break
                    elif buy[cnt]:
                        mcover[cnt] = 1
                        break

        return mcover

    @numba.jit
    def apply_stops_sell(self, buy, short, buyprice, stoploss, takeprofit):
        ''' apply stops on long trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        buyprice = buyprice.values
        nnn = len(buy)
        msell = np.zeros(nnn, dtype='int64')
        _open = self.ohlc.open.values

        for i in xrange(nnn):
            if buy[i]:
                topen = buyprice[i]
                for cnt in xrange(i+1, nnn, 1):
                    val = _open[cnt] - topen
                    if val > takeprofit * tsize:
                        msell[cnt] = 1
                        break
                    elif val < -stoploss * tsize:
                        msell[cnt] = 1
                        break
                    elif short[cnt]:
                        msell[cnt] = 1
                        break

        return msell


    def run(self, buy, short, sell, cover, buyprice, shortprice, sellprice, coverprice):
        ''' calculate equity based on trade signals '''
        idx = ((buy > 0) | (short > 0) | (sell > 0) | (cover > 0))

        buy = buy[idx]
        short = short[idx]
        sell = sell[idx]
        cover = cover[idx]
        buyprice = buyprice[idx]
        shortprice = shortprice[idx]
        sellprice = sellprice[idx]
        coverprice = coverprice[idx]

        myeq = self.starting_equity
        myequity = np.empty(len(buy))
        imp_equity = np.empty(len(buy))
        myequity.fill(myeq)
        imp_equity.fill(myeq)
        mytrades = []
        mvalue = np.zeros(1, dtype='float64')

        for i, item in enumerate(zip(buy.values, short.values, sell.values, cover.values,
                                     buyprice.values, shortprice.values, sellprice.values,
                                     coverprice.values)):

            if item[1] > 0: # *** active short *** #

                if self.risk == 0.0:
                    lot_size = 1
                else:
                    lot_size = int(myequity[i] / self.margin_required * self.risk)

                imp_equity[i] = myequity[i]
                loceq = myequity[i]
                mytrades.append({'index': short.index[i], 'direction': 'short',
                                 'lotsize': -lot_size, 'price': shortprice[i],
                                 'value': 0, 'equity': myequity[i], 'ticks': 0})

                for cnt, col1, col2, col3 in izip(count(), cover.values[i+1:],
                                                  coverprice.values[i+1:], buy.values[i+1:]):

                    trd_ticks = (item[5] - col2) / self.tick_size
                    trd_val = trd_ticks * self.tickvalue * lot_size
                    imp_equity[cnt+i+1] = loceq + trd_val

                    if (col1 == item[1]) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        mvalue = np.append(mvalue, trd_val)

                        value = myeq + np.sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value

                        mytrades.append({'index': cover.index[cnt+i+1], 'direction': 'cover',
                                         'lotsize': lot_size, 'price': coverprice[cnt+i+1],
                                         'value': trd_val, 'equity': value, 'ticks': trd_ticks})

                        break

            elif item[0] > 0: # *** active long *** #

                if self.risk == 0.0:
                    lot_size = 1
                else:
                    lot_size = int(myequity[i] / self.margin_required * self.risk)

                imp_equity[i] = myequity[i]
                loceq = myequity[i]

                mytrades.append({'index': buy.index[i], 'direction': 'buy',
                                 'lotsize': lot_size, 'price': buyprice[i],
                                 'value': 0, 'equity': myequity[i], 'ticks': 0})

                for cnt, col1, col2, col3 in izip(count(), sell.values[i+1:],
                                                  sellprice.values[i+1:], short.values[i+1:]):

                    trd_ticks = (col2 - item[4]) / self.tick_size
                    trd_val = trd_ticks * self.tickvalue * lot_size
                    imp_equity[cnt+i+1] = loceq + trd_val

                    if (col1 > 0) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        mvalue = np.append(mvalue, trd_val)

                        value = myeq + np.sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value

                        mytrades.append({'index': sell.index[cnt+i+1], 'direction': 'sell',
                                         'lotsize': -lot_size, 'price': sellprice[cnt+i+1],
                                         'value': trd_val, 'equity': value, 'ticks': trd_ticks})

                        break

        mytrades = pd.DataFrame(mytrades).dropna()
        mytrades.set_index('index', drop=True, append=False, inplace=True, verify_integrity=False)
        mytrades.index = pd.to_datetime(mytrades.index)
        mytrades['symbol'] = self.symbol
        mytrades = mytrades[:][['symbol', 'direction', 'lotsize', 'price',
                                'value', 'equity', 'ticks']]
        mytrades = mytrades.sort_index()
        self.trades = mytrades
        self.imp_equity = imp_equity
        #mytrades.to_csv('trades.csv')


    def analyze_results(self, rfr):
        ''' analyze trades '''
        print 'Starting Equity: ' + str(self.trades['equity'][0])
        print 'Final Equity: ' + str(self.trades['equity'][-1])
        pt_count = 0.0
        pt_short = 0.0
        t_short = 0.0
        t_long = 0.0
        pt_long = 0.0
        trades = self.trades
        total_trades = len(trades)/2
        for i in xrange(len(trades)):
            if trades['direction'][i] == 'short':
                t_short += 1
            if trades['direction'][i] == 'buy':
                t_long += 1
            if trades['value'][i] > 0:
                pt_count += 1
                if trades['direction'][i] == 'cover':
                    pt_short += 1
                if trades['direction'][i] == 'sell':
                    pt_long += 1

        print 'Profitable trades: '+str(int(pt_count))
        print 'Losing trades: '+str(int(total_trades - pt_count))
        if total_trades > 0:
            winners = pt_count / total_trades*100
            print 'Winrate: ' + str(round(winners, 2)) + '%'
        else:
            print 'No trades made'

        if t_short > 0:
            print 'Short winrate: ' + str(round(pt_short/t_short*100, 2)) + '%'
        if t_long > 0:
            print 'Long winrate: ' + str(round(pt_long/t_long*100, 2)) + '%'


        daily_ret = self.trades.equity.resample('1D').last().dropna().pct_change()
        daily_excess = daily_ret - rfr/252
        sharpe = np.sqrt(252) * daily_excess.mean() / daily_excess.std()
        print 'Sharpe: {:.2f} '.format(float(sharpe))

        sortino = np.sqrt(252) * daily_excess.mean() / daily_excess[daily_excess < 0].std()
        print 'Sortino: {:.2f} '.format(float(sortino))

        period = (self.ohlc.index[-1] - self.ohlc.index[0]).total_seconds() / (31557600)
        cagr = (self.trades.equity.values[-1] / self.trades.equity.values[0]) ** (1/float(period))-1
        print 'CAGR: {:.2%} '.format(float(cagr))

        tval = self.trades.value.values
        pfr = tval[tval > 0].sum() / abs(tval[tval < 0].sum())
        print 'Profit factor: {:.2f}'.format(pfr)

        new_equity = self.trades.equity[(self.trades.equity != self.trades.equity.shift(1))]
        rolling_dd = new_equity.rolling(min_periods=0, window=10,
                                        center=False).apply(func=_max_rolling_dd)

        zipp = zip(new_equity, rolling_dd)
        df1 = pd.DataFrame(zipp, index=new_equity.index)
        df1.columns = ['Equity', 'Drawdown']

        maxdd, closs = max_draw(trades)
        print 'Consecutive losses: ' + str(closs)
        print 'Max drawdown: ' + str(round(maxdd, 2))+'%'

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        df1.plot(ax=ax1)
        ax1.set_ylabel('Portfolio value (USD)')
        ax1.set_xlabel('')
        plt.gcf().set_size_inches(8, 10)
        plt.show()

    def plot_trades(self, startdate, enddate):
        ''' plot trades '''
        if len(self.trades > 0):
            trd = self.trades
            subset = slice(str(startdate), str(enddate))
            frm = trd.ix[subset]

            lent = frm.price[(frm.direction == 'buy') & (frm.lotsize > 0)]
            sent = frm.price[(frm.direction == 'short') & (frm.lotsize < 0)]
            lex = frm.price[(frm.direction == 'sell') & (frm.lotsize < 0)]
            sex = frm.price[(frm.direction == 'cover') & (frm.lotsize > 0)]

            if len(lent > 0):
                pylab.plot(lent.index, lent.values, '^', color='lime', markersize=12,
                           label='long enter')
            if len(sent > 0):
                pylab.plot(sent.index, sent.values, 'v', color='red', markersize=12,
                           label='short enter')
            if len(lex > 0):
                pylab.plot(lex.index, lex.values, 'o', color='lime', markersize=7,
                           label='long exit')
            if len(sex > 0):
                pylab.plot(sex.index, sex.values, 'o', color='red', markersize=7,
                           label='short exit')


            self.ohlc.open.ix[subset].plot(color='black', label='price')
            eqt = pd.DataFrame(self.trades.ticks[subset].cumsum()*self.tick_size)
            eqt.columns = ['value']
            idx = eqt.index

            (eqt + self.ohlc.open[idx[0]]).plot(color='red', style='-', label='value')
            self.ohlc.close.ix[subset].plot(color='black', label='price')
        else:
            print 'No trades to plot!'


    def annual_gains(self, start, end):
        ''' calculate annual gains '''
        gains = []
        years = []

        if self.trades.lotsize.abs().mean() == 1:
            for i in xrange(start, end+1, 1):
                gain = (self.trades[str(i)].equity[-1] - \
                                    self.trades[str(i)].equity[0])/self.trades.equity[0]
                gains.append(gain*100)
                years.append(i)
        else:
            for i in xrange(start, end+1, 1):
                gain = (self.trades[str(i)].equity[-1] - \
                                    self.trades[str(i)].equity[0])/self.trades[str(i)].equity[0]
                gains.append(gain*100)
                years.append(i)

        _mean = pd.Series(gains).mean()
        zipp = zip(gains, years)
        df1 = pd.DataFrame(zipp)

        df1.columns = ['gains', 'years']
        df1['mean'] = _mean

        fig = plt.figure()
        ax1 = fig.add_subplot(212)
        df1.plot(x='years', y='gains', ax=ax1, kind='bar', color='green')
        ax1.set_ylabel('Annual Returns (%)')
        ax1.set_xlabel('')
        plt.gcf().set_size_inches(8, 10)
        plt.show()

    def analyze_results_ffn(self, rfr):
        ''' analyze performance with ffn'''
        data = self.trades.equity.resample('1D').last().dropna()
        myffn = PerformanceStats(data, rfr)
        myffn.display()
        print '\n'
        myffn.display_monthly_returns()
        print '\n'
        #print myffn.stats
