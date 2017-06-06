#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:26:03 2017
@author: https://github.com/Quantmatic/
"""
import time
from itertools import izip, count
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient


global TRADES
global TICKS
global OHLC
global TICK_SIZE
global EQUITY
global IMP_EQUITY


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

def ex_rem(array1, array2, maxtrades=1):
    """ Remove excessive signals """
    #start = time.time()
    assert array1.index.equals(array2.index), 'Indices do not match'
    output = pd.Series(False, dtype=bool, index=array1.index)
    iterator = 0

    for i, item in enumerate(zip(array1, array2)):
        if item[0] and i >= iterator:
            output[i] = True
            cnt = 1
            for j, arr1, arr2 in izip(count(), array1[i+1:], array2[i+1:]):
                if arr1:
                    cnt += 1
                    if cnt > maxtrades:
                        output[j+i+1] = False
                    else:
                        output[j+i+1] = True

                if arr2:
                    iterator = i+j+1
                    break
    #print 'ExRem processed in ' + str(time.time() - start) + ' seconds'
    return output

# plot_trades('2015','2015')
def plot_trades(startdate, enddate):
    ''' plot trades '''
    if len(TRADES > 0):
        trd = TRADES
        subset = slice(str(startdate), str(enddate))
        frm = trd.ix[subset]

        lent = frm.price[(frm.direction == 'buy') & (frm.lotsize > 0)]
        sent = frm.price[(frm.direction == 'short') & (frm.lotsize < 0)]
        lex = frm.price[(frm.direction == 'sell') & (frm.lotsize < 0)]
        sex = frm.price[(frm.direction == 'cover') & (frm.lotsize > 0)]

        import matplotlib.pylab as pylab
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

        global OHLC
        OHLC.open.ix[subset].plot(color='black', label='price')
        global TICK_SIZE
        eqt = TICKS.ix[subset].cumsum()*TICK_SIZE
        idx = eqt.index
        (eqt + OHLC.open[idx[0]]).plot(color='red', style='-')
        OHLC.open.ix[idx[0]:idx[-1]].plot(color='black', label='price')
        OHLC.open.ix[subset].plot(color='black', label='price')
    else:
        print 'No trades to plot!'


def annual_gains(start, end):
    ''' calculate annual gains '''
    gains = []
    years = []

    if TRADES.lotsize.abs().mean() == 1:
        for i in xrange(start, end+1, 1):
            gain = (TRADES[str(i)].equity[-1] - TRADES[str(i)].equity[0])/EQUITY[0]
            gains.append(gain*100)
            years.append(i)
    else:
        for i in xrange(start, end+1, 1):
            gain = (TRADES[str(i)].equity[-1] - TRADES[str(i)].equity[0])/TRADES[str(i)].equity[0]
            gains.append(gain*100)
            years.append(i)

    #print 'Average annual gain ', round(pd.Series(gains).mean(), 2), '%'
    _mean = pd.Series(gains).mean()
    zipp = zip(gains, years)
    df1 = pd.DataFrame(zipp)

    df1.columns = ['gains', 'years']
    df1['mean'] = _mean

    fig = plt.figure()
    ax1 = fig.add_subplot(212)
    df1.plot(x='years', y='gains', ax=ax1, kind='bar', color='green')
    #df1['mean'].plot(y='mean', ax=ax1, color='cyan', kind='line')
    ax1.set_ylabel('Annual Returns (%)')
    ax1.set_xlabel('')
    plt.gcf().set_size_inches(8, 10)
    plt.show()

def max_draw(trades):
    ''' calculate draw down '''
    maxeq = 0
    maxdd = 0 #maxdd
    closs = 0 #consecutive losses
    trades = trades.equity.sort_index()
    trades = trades[trades != trades.shift(1)]
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

class Amipy(object):
    """ initialize constants required for backtest """
    def __init__(self, SYMBOL, EQUITY, MARGIN, TICKVALUE, TICKSIZE, RISK, DATA):
        self.symbol = SYMBOL
        self.starting_equity = EQUITY
        self.margin_required = MARGIN
        self.tickvalue = TICKVALUE
        self.tick_size = TICKSIZE
        self.risk = RISK
        global OHLC
        OHLC = DATA
        global TICK_SIZE
        TICK_SIZE = TICKSIZE

    def apply_stops_cover(self, buy, short, shortprice, stoploss, takeprofit):
        ''' apply stops on short trades '''
        #start = time.time()
        tsize = self.tick_size
        mcover = [False for i in xrange(len(short))]
        _open = OHLC.open

        for i, item in enumerate(zip(short, shortprice)):

            if item[0] > 0: #*** short ***#
                topen = item[1]
                for cnt, col, price in izip(count(), buy[i+1:], _open[i+1:]):
                    val = topen - price
                    if val > takeprofit * tsize:
                        mcover[cnt+i+1] = True
                        break
                    elif val < -stoploss * tsize:
                        mcover[cnt+i+1] = True
                        break
                    elif col:
                        mcover[cnt+i+1] = True
                        break

        #print '_ApplyStopCover processed in ' + str(time.time() - start) + ' seconds'
        return mcover


    def apply_stops_sell(self, buy, short, buyprice, stoploss, takeprofit):
        ''' apply stops on long trades '''
        #start = time.time()
        tsize = self.tick_size
        msell = [False for i in xrange(len(short))]
        _open = OHLC.open

        for i, col in enumerate(zip(buy, buyprice)):
            if col[0] > 0: #*** active long ***#
                topen = col[1]
                for j, item, price in izip(count(), short[i+1:], _open[i+1:]):
                    val = price - topen
                    if val > takeprofit * tsize:
                        msell[j+i+1] = True
                        break
                    elif val < -stoploss * tsize:
                        msell[j+i+1] = True
                        break
                    elif item:
                        msell[j+i+1] = True
                        break

        #print '_ApplyStopSell processed in ' + str(time.time() - start) + ' seconds'
        return msell

    def run(self, buy, short, sell, cover, buyprice, shortprice, sellprice, coverprice):
        ''' calculate equity based on trade signals '''
        #mtimer = time.time()
        idx = ((buy > 0) | (short > 0) | (sell > 0) | (cover > 0))

        buy = buy[idx]
        short = short[idx]
        sell = sell[idx]
        cover = cover[idx]
        buyprice = buyprice[idx]
        shortprice = shortprice[idx]
        sellprice = sellprice[idx]
        coverprice = coverprice[idx]

        myticks = []
        myeq = self.starting_equity
        myequity = pd.Series(myeq, index=buy.index, name='equity')
        mytrades = []
        mvalue = []
        imp_equity = pd.Series(myeq, index=buy.index, name='imp_equity')

        for i, item in enumerate(zip(buy, short, sell, cover, buyprice,
                                     shortprice, sellprice, coverprice)):

            if item[1] > 0: # *** active short *** #

                if self.risk == 0.0:
                    lot_size = 1
                else:
                    lot_size = int(myequity[i] / self.margin_required * self.risk)

                imp_equity[i] = myequity[i]
                loceq = myequity[i]
                mytrades.append({'index': short.index[i], 'direction': 'short',
                                 'lotsize': -lot_size, 'price': shortprice[i],
                                 'value': 0, 'equity': myequity[i]})

                for cnt, col1, col2, col3 in izip(count(), cover[i+1:],
                                                  coverprice[i+1:], buy[i+1:]):

                    trd_ticks = (item[5] - col2) / self.tick_size
                    trd_val = trd_ticks * self.tickvalue * lot_size
                    imp_equity[cnt+i+1] = loceq + trd_val

                    if (col1 == item[1]) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        myticks.append({'index': cover.index[cnt+i+1], 'value': trd_ticks})
                        mvalue.append(trd_val)

                        value = myeq + sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value

                        mytrades.append({'index': cover.index[cnt+i+1], 'direction': 'cover',
                                         'lotsize': lot_size, 'price': coverprice[cnt+i+1],
                                         'value': trd_val, 'equity': value})

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
                                 'value': 0, 'equity': myequity[i]})

                for cnt, col1, col2, col3 in izip(count(), sell[i+1:],
                                                  sellprice[i+1:], short[i+1:]):

                    trd_ticks = (col2 - item[4]) / self.tick_size
                    trd_val = trd_ticks * self.tickvalue * lot_size
                    imp_equity[cnt+i+1] = loceq + trd_val

                    if (col1 > 0) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        myticks.append({'index': sell.index[cnt+i+1], 'value': trd_ticks})
                        mvalue.append(trd_val)

                        value = myeq + sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value

                        mytrades.append({'index': sell.index[cnt+i+1], 'direction': 'sell',
                                         'lotsize': -lot_size, 'price': sellprice[cnt+i+1],
                                         'value': trd_val, 'equity': value})

                        break

        #print 'Equity processed in ' + str(time.time()-mtimer) + ' seconds'
        global TRADES
        mytrades = pd.DataFrame(mytrades).dropna()
        mytrades.set_index('index', drop=True, append=False, inplace=True, verify_integrity=False)
        mytrades.index = pd.to_datetime(mytrades.index)
        mytrades['symbol'] = self.symbol
        TRADES = mytrades[:][['symbol', 'direction', 'lotsize', 'price', 'value', 'equity']]
        TRADES = TRADES.sort_index()
        #TRADES.to_csv('trades.csv')
        global TICKS
        TICKS = pd.DataFrame(myticks).dropna()
        TICKS.set_index('index', drop=True, append=False, inplace=True, verify_integrity=False)
        TICKS.index = pd.to_datetime(TICKS.index)
        global EQUITY
        EQUITY = myequity
        global IMP_EQUITY
        IMP_EQUITY = imp_equity


    def _max_dd(self, ser):
        ''' max dd calculations '''
        max2here = pd.Series(ser).expanding().max()
        dd2here = ser - max2here
        return dd2here.min()

    def analize_results(self):
        ''' analize trades '''
        pt_count = 0.0
        pt_short = 0.0
        t_short = 0.0
        t_long = 0.0
        pt_long = 0.0
        total_trades = len(TRADES)/2
        for i in xrange(len(TRADES)):
            if TRADES['direction'][i] == 'short':
                t_short += 1
            if TRADES['direction'][i] == 'buy':
                t_long += 1
            if TRADES['value'][i] > 0:
                pt_count += 1
                if TRADES['direction'][i] == 'cover':
                    pt_short += 1
                if TRADES['direction'][i] == 'sell':
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

        _days = lambda eqd: eqd.resample('D').sum().dropna()

        day = _days(TICKS)
        annual_sharpe = (day.mean() / day.std()) * (252*0.05)
        print 'Sharpe: ' + str(round(annual_sharpe, 2))

        annual_sortino = (day.mean() / day[day < day.shift()].std()) * (252*0.05)
        print 'Sortino: ' + str(round(annual_sortino, 2))

        years = TRADES.index.year
        period = (years[-1] - years[0]) + 1
        cagr = int((float(TRADES.equity[-1]) / float(TRADES.equity[0])))**(1/float(period))-1
        print 'CAGR: {:.2%} '.format(float(cagr))

        pfr = lambda eqd: abs(eqd[eqd > 0].sum() / eqd[eqd < 0].sum())
        print 'Profit factor: {:.2}'.format(pfr(TRADES.value))

        new_equity = TRADES.equity[(TRADES.equity != TRADES.equity.shift(1))]
        rolling_dd = new_equity.rolling(min_periods=0, window=10,
                                        center=False).apply(func=self._max_dd)

        zipp = zip(new_equity, rolling_dd)
        df1 = pd.DataFrame(zipp, index=new_equity.index)
        df1.columns = ['Equity', 'Drawdown']

        maxdd, closs = max_draw(TRADES)
        print 'Consecutive losses: ' + str(closs)
        print 'Max drawdown: ' + str(round(maxdd, 2))+'%'

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        df1[:].plot(ax=ax1)
        ax1.set_ylabel('Portfolio value (USD)')
        ax1.set_xlabel('')
        plt.gcf().set_size_inches(8, 10)
        plt.show()
