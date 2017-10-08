#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#pylint: disable=no-member
"""
Created on Tue May 16 11:26:03 2017
@author: github.com/Quantmatic
"""
from __future__ import print_function, division
from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pymongo import MongoClient
from ffn import PerformanceStats
#import numba


def df_resample(dframe, interval):
    """ resample 1m data """
    _result = dframe.resample(interval).agg({'open': 'first',
                                             'high': 'max',
                                             'low': 'min',
                                             'close': 'last',
                                             'volume': 'sum'})

    _df = pd.DataFrame(_result).dropna(axis=0)

    _df[['volume']] = _df[['volume']].fillna(0.0).astype(int)
    return _df[:][['open', 'high', 'low', 'close', 'volume']]


def mongo_grab(symbol, dbname, startdate, enddate, interval='60min', resample=False):
    ''' Grab the required stretch of quotes from MongoDB '''
    client = MongoClient()
    dbase = client[dbname]
    collections = dbase.collection_names()
    if symbol in collections:
        collection = dbase[symbol]
        cursor = collection.find({'datetime': {'$gte': startdate, '$lt': enddate}})
        data = list(cursor)
        _df = pd.DataFrame(data)
        _df.columns = _df.columns.str.lower()
        _df.set_index('datetime', drop=False, append=False, inplace=True, verify_integrity=False)
        _df.index = pd.to_datetime(_df.index)
        _df = _df[:][['open', 'high', 'low', 'close', 'volume']]
        if resample:
            resampled_df = df_resample(_df, interval)
            return resampled_df

        return _df
    else:
        print('Error! Symbol not found in db! '+symbol)
        return -1


#@numba.jit
def __remove(array1, array2, maxtrades=1):
    """ Remove excessive signals """
    nnn = len(array1)
    output = np.zeros(nnn, dtype='int64')
    i = 0

    while i < nnn:
        if array1[i]:
            output[i] = 1
            cnt = 1
            j = i+1
            while j < nnn:
                if array1[j]:
                    cnt += 1
                    if cnt > maxtrades:
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


def _max_draw(equity):
    ''' calculate drawdown '''
    return (equity / equity.expanding(min_periods=1).max()).min() - 1


def _consecutive_loss(equity):
    ''' calculate consecutive losses '''
    equity = equity[equity != equity.shift(1)]
    temp = (equity < equity.shift(1)).astype(int)
    closs = temp.groupby((temp != temp.shift(1)).cumsum()).cumsum().max()
    return closs


def _max_rolling_dd(ser):
    ''' max dd calculations '''
    max2here = pd.Series(ser).expanding().max()
    dd2here = ser - max2here
    return dd2here.min()


def _plot(_df, subplot=211, ylabel='', xlabel='', ysize=12, xsize=15, legend=False, title=''):
    ''' make a plot '''
    fig = plt.figure()
    ax1 = fig.add_subplot(subplot)
    _df.plot(ax=ax1)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.legend().set_visible(legend)
    plt.gcf().set_size_inches(ysize, xsize)
    plt.title(title)
    plt.show()
    plt.close()


def adjust_quotes(ohlc):
    ''' remove stock splits for smooth backtests '''
    i = len(ohlc)-1
    data = ohlc.copy()
    _open = ohlc.open.values
    close = ohlc.close.values
    temp = ['open', 'high', 'low', 'close']
    while i > 1:
        if _open[i] < _open[i-1]*0.8:
            data.loc[ohlc.index[i:], temp] += close[i-1]-_open[i]

        i -= 1

    return data


def analyze_portfolio(portfolio, rfr, plot=True):
    ''' analyze portfolio '''
    portfolio = portfolio.fillna(method='ffill').fillna(method='bfill')
    portfolio['Total'] = portfolio.sum(axis=1)
    total_ret = portfolio.Total[-1] / portfolio.Total[0] - 1
    daily_ret = portfolio.Total.resample('1D').last().dropna().pct_change()
    daily_excess = daily_ret - rfr/252
    sharpe = 252**0.5 * daily_excess.mean() / daily_excess.std()
    sortino = 252**0.5 * daily_excess.mean() / daily_excess[daily_excess < 0].std()
    period = (portfolio.index[-1] - portfolio.index[0]).total_seconds() / (31557600)
    cagr = (portfolio.Total.values[-1] / portfolio.Total.values[0]) ** (1/float(period))-1
    maxdd = _max_draw(portfolio.Total)

    print('\n')
    print('Starting Portfolio Equity: ' + str(portfolio['Total'][0]))
    print('Final Portfolio Equity: ' + str(portfolio['Total'][-1]))
    print('Total return: {:.2%} '.format(float(total_ret)))
    print('Daily return: {:.2%} '.format(float(daily_ret.mean())))
    print('Risk: {:.2%} '.format(float(daily_ret.std())))
    print('Sharpe: {:.2f} '.format(float(sharpe)))
    print('Sortino: {:.2f} '.format(float(sortino)))
    print('CAGR: {:.2%} '.format(float(cagr)))
    print('Max drawdown: {:.2%} '.format(float(maxdd)))

    if plot:
        new_equity = portfolio.Total[(portfolio.Total != portfolio.Total.shift(1))]
        rolling_dd = new_equity.rolling(min_periods=1, window=2,
                                        center=False).apply(func=_max_rolling_dd)

        zipp = list(zip(new_equity, rolling_dd))
        df1 = pd.DataFrame(zipp, index=new_equity.index)
        df1.columns = ['Equity', 'Drawdown']

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        df1.plot(ax=ax1)
        ax1.set_ylabel('Portfolio value (USD)')
        ax1.set_xlabel('')
        plt.gcf().set_size_inches(12, 15)
        plt.show()
        plt.close()

    return {'total_ret': total_ret, 'sharpe': sharpe, 'sortino': sortino,
            'cagr': cagr, 'maxdd': maxdd}


def analyze_portfolio_ffn(portfolio, rfr):
    ''' analyze portfolio with ffn'''
    portfolio = portfolio.fillna(method='ffill').fillna(method='bfill')
    portfolio['Total'] = portfolio.sum(axis=1)
    myffn = PerformanceStats(portfolio.Total, rfr)
    print('\n')
    myffn.display()
    print('\n')
    myffn.display_monthly_returns()
    print('\n')
    #print(myffn.stats)


class Amipy(object):
    """ initialize constants required for backtest """
    def __init__(self, CONTEXT, DATA):
        self.__dict__.update(CONTEXT.__dict__)
        self.trades = None
        self.imp_equity = None
        self.ohlc = DATA
        self.stats = {}
        self.margin = 0
        self.equity = 0

    #@numba.jit
    def apply_stops_cover(self, buy, short, shortprice, stoploss, takeprofit):
        ''' apply tick based stops on short trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        shortprice = shortprice.values
        _open = self.ohlc.open.values
        mcover = np.zeros(len(short), dtype='int64')

        nnn = len(buy)
        for i in range(nnn):
            if short[i]:
                topen = shortprice[i]
                for cnt in range(i+1, nnn, 1):
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

    #@numba.jit
    def apply_stops_sell(self, buy, short, buyprice, stoploss, takeprofit):
        ''' apply tick based stops on long trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        buyprice = buyprice.values
        nnn = len(buy)
        msell = np.zeros(nnn, dtype='int64')
        _open = self.ohlc.open.values

        for i in range(nnn):
            if buy[i]:
                topen = buyprice[i]
                for cnt in range(i+1, nnn, 1):
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

    #@numba.jit
    def apply_stops_cover_rq(self, buy, short, shortprice, stoploss, takeprofit):
        ''' apply rolling quantile stops on short trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        shortprice = shortprice.values
        _open = self.ohlc.open.values
        mcover = np.zeros(len(short), dtype='int64')

        nnn = len(buy)
        for i in range(nnn):
            if short[i]:
                topen = shortprice[i]
                for cnt in range(i+1, nnn, 1):
                    val = topen - _open[cnt]
                    if val > takeprofit[i] * tsize:
                        mcover[cnt] = 1
                        break
                    elif val < -stoploss[i] * tsize:
                        mcover[cnt] = 1
                        break
                    elif buy[cnt]:
                        mcover[cnt] = 1
                        break

        return mcover

    #@numba.jit
    def apply_stops_sell_rq(self, buy, short, buyprice, stoploss, takeprofit):
        ''' apply rolling quantile stops on long trades '''
        tsize = self.tick_size
        short = short.values
        buy = buy.values
        buyprice = buyprice.values
        nnn = len(buy)
        msell = np.zeros(nnn, dtype='int64')
        _open = self.ohlc.open.values

        for i in range(nnn):
            if buy[i]:
                topen = buyprice[i]
                for cnt in range(i+1, nnn, 1):
                    val = _open[cnt] - topen
                    if val > takeprofit[i] * tsize:
                        msell[cnt] = 1
                        break
                    elif val < -stoploss[i] * tsize:
                        msell[cnt] = 1
                        break
                    elif short[cnt]:
                        msell[cnt] = 1
                        break

        return msell

    #@numba.jit
    def apply_stops_cover_pct(self, buy, short, shortprice, stoploss, takeprofit):
        ''' apply percentage based stops on short trades '''
        short = short.values
        buy = buy.values
        nnn = len(buy)
        shortprice = shortprice.values
        mcover = np.zeros(nnn, dtype='int64')
        _open = self.ohlc.open.values


        for i in range(nnn):
            if short[i]:
                topen = shortprice[i]

                for cnt in range(i+1, nnn, 1):
                    val = topen - _open[cnt]
                    if val/topen > takeprofit/100.0: #% gain
                        mcover[cnt] = 1
                        break
                    elif val/topen < -stoploss/100.0:
                        mcover[cnt] = 1
                        break
                    elif buy[cnt]: #opposite signal
                        mcover[cnt] = 1
                        break

        return mcover

    #@numba.jit
    def apply_stops_sell_pct(self, buy, short, buyprice, stoploss, takeprofit):
        ''' apply percentage based stops on long trades '''
        short = short.values
        buy = buy.values
        buyprice = buyprice.values
        nnn = len(buy)
        msell = np.zeros(nnn, dtype='int64')
        _open = self.ohlc.open.values

        for i in range(nnn):
            if buy[i]:
                topen = buyprice[i]
                for cnt in range(i+1, nnn, 1):
                    val = _open[cnt] - topen
                    if val/topen > takeprofit/100.0: #% gain
                        msell[cnt] = 1
                        break
                    elif val/topen < -stoploss/100.0:
                        msell[cnt] = 1
                        break
                    elif short[cnt]: #opposite signal
                        msell[cnt] = 1
                        break

        return msell

    def run(self, buy, short, sell, cover, buyprice, shortprice, sellprice, coverprice):
        ''' calculate equity based on trade signals '''
        #idx = ((buy > 0) | (short > 0) | (sell > 0) | (cover > 0))
        idx=buy.index

        buy = buy[idx]
        short = short[idx]
        sell = sell[idx]
        cover = cover[idx]
        buyprice = buyprice[idx]
        shortprice = shortprice[idx]
        sellprice = sellprice[idx]
        coverprice = coverprice[idx]
        _open = self.ohlc.open[idx].values

        myeq = self.starting_equity
        myequity = np.empty(len(buy))
        imp_equity = np.empty(len(buy))
        imargin = np.zeros(len(buy))
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
                    umargin = self.margin_required  # used margin
                    if self.margin_required == 0:
                        umargin = _open[i]
                elif self.margin_required == 0:
                    lot_size = int(myequity[i] / _open[i] * self.risk)
                    umargin = _open[i] * lot_size
                else:
                    lot_size = int(myequity[i] / self.margin_required * self.risk)
                    umargin = lot_size * self.margin_required

                if lot_size < 0 or myequity[i] < 0:
                    continue

                imargin[i] = imargin[i] + umargin
                imp_equity[i] = myequity[i]
                loceq = myequity[i]

                mytrades.append({'index': short.index[i], 'direction': 'short',
                                 'lotsize': -lot_size, 'price': shortprice[i],
                                 'value': 0, 'equity': myequity[i], 'ticks': 0,
                                 'umargin': umargin})

                for cnt, col1, col2, col3 in zip(count(), cover.values[i+1:],
                                                 coverprice.values[i+1:], buy.values[i+1:]):

                    trd_ticks = (item[5] - col2) / self.tick_size
                    commission = self.commission * lot_size
                    trd_val = (trd_ticks * self.tick_value * lot_size) - commission
                    imp_equity[cnt+i+1] = loceq + trd_val
                    imargin[cnt+i+1] = imargin[cnt+i]

                    if (col1 == item[1]) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        elif self.margin_required == 0:
                            lot_size = int(myequity[i] / _open[i] * self.risk)
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        mvalue = np.append(mvalue, trd_val)

                        value = myeq + np.sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value
                        imargin[cnt+i+1] = 0

                        mytrades.append({'index': cover.index[cnt+i+1], 'direction': 'cover',
                                         'lotsize': lot_size, 'price': coverprice[cnt+i+1],
                                         'value': trd_val, 'equity': value, 'ticks': trd_ticks,
                                         'umargin': 0})

                        break

            elif item[0] > 0: # *** active long *** #

                if self.risk == 0.0:
                    lot_size = 1
                    umargin = self.margin_required  # used margin
                    if self.margin_required == 0:
                        umargin = _open[i]
                elif self.margin_required == 0:
                    lot_size = int(myequity[i] / _open[i] * self.risk)
                    umargin = _open[i] * lot_size
                else:
                    lot_size = int(myequity[i] / self.margin_required * self.risk)
                    umargin = lot_size * self.margin_required

                if lot_size < 0 or myequity[i] < 0:
                    continue

                imargin[i] = imargin[i] + umargin
                imp_equity[i] = myequity[i]
                loceq = myequity[i]

                mytrades.append({'index': buy.index[i], 'direction': 'buy',
                                 'lotsize': lot_size, 'price': buyprice[i],
                                 'value': 0, 'equity': myequity[i], 'ticks': 0,
                                 'umargin': umargin})

                for cnt, col1, col2, col3 in zip(count(), sell.values[i+1:],
                                                 sellprice.values[i+1:], short.values[i+1:]):

                    trd_ticks = (col2 - item[4]) / self.tick_size
                    commission = self.commission * lot_size
                    trd_val = (trd_ticks * self.tick_value * lot_size) - commission
                    imp_equity[cnt+i+1] = loceq + trd_val
                    imargin[cnt+i+1] = imargin[cnt+i]

                    if (col1 > 0) | (col3 > 0):

                        if self.risk == 0.0:
                            lot_size = 1
                        elif self.margin_required == 0:
                            lot_size = int(myequity[i] / _open[i] * self.risk)
                        else:
                            lot_size = int(myequity[i] / self.margin_required * self.risk)

                        mvalue = np.append(mvalue, trd_val)

                        value = myeq + np.sum(mvalue)
                        myequity[cnt+i+1:] = value
                        imp_equity[cnt+i+1:] = value
                        imargin[cnt+i+1] = 0

                        mytrades.append({'index': sell.index[cnt+i+1], 'direction': 'sell',
                                         'lotsize': -lot_size, 'price': sellprice[cnt+i+1],
                                         'value': trd_val, 'equity': value, 'ticks': trd_ticks,
                                         'umargin': 0})

                        break

        mytrades = pd.DataFrame(mytrades).dropna()
        mytrades.set_index('index', drop=True, append=False, inplace=True, verify_integrity=False)
        mytrades.index = pd.to_datetime(mytrades.index)
        mytrades['symbol'] = self.symbol
        mytrades = mytrades[:][['symbol', 'direction', 'lotsize', 'price',
                                'value', 'equity', 'ticks', 'umargin']]
        mytrades = mytrades.sort_index()
        self.trades = mytrades
        self.imp_equity = imp_equity
        self.equity = myequity
        self.margin = imargin
        #mytrades.to_csv('trades.csv')

    def analyze_results_silent(self, rfr):
        ''' analyze trades '''
        trades = self.trades
        total_ret = trades.equity[-1] / trades.equity[0] - 1
        self.stats['total_ret'] = round(total_ret, 3)
        temp = trades.equity.resample('1D').last().dropna()
        temp.name = self.symbol
        idx = self.ohlc.close.resample('1D').last().index
        equity = pd.DataFrame(temp, index=idx).fillna(method='ffill').fillna(method='bfill')
        daily_ret = equity[self.symbol].pct_change()

        self.stats['symbol'] = self.symbol
        self.stats['daily_return'] = round(daily_ret.mean(), 4)
        self.stats['daily_risk'] = round(daily_ret.std(), 4)
        daily_excess = daily_ret - rfr/252
        sharpe = 252**0.5 * daily_excess.mean() / daily_excess.std()
        self.stats['sharpe'] = round(sharpe, 3)
        sortino = 252**0.5 * daily_excess.mean() / daily_excess[daily_excess < 0].std()
        self.stats['sortino'] = round(sortino, 3)
        period = (self.ohlc.index[-1] - self.ohlc.index[0]).total_seconds() / (31557600)
        cagr = (trades.equity.values[-1] / trades.equity.values[0]) ** (1/float(period))-1
        self.stats['cagr'] = round(cagr, 3)
        tval = trades.value.values
        pfr = tval[tval > 0].sum() / max(abs(tval[tval < 0].sum()), 1)
        self.stats['pfr'] = round(pfr, 3)
        maxdd = _max_draw(trades.equity.sort_index())
        self.stats['maxdd'] = round(maxdd, 3)

    def analyze_results(self, rfr, plot=True):
        ''' analyze trades '''
        trades = self.trades
        total_ret = trades.equity[-1] / trades.equity[0] - 1
        self.stats['total_ret'] = round(total_ret, 3)
        t_short = len(trades[trades.direction == 'short'])
        t_long = len(trades[trades.direction == 'buy'])
        pt_count = len(trades[trades.value > 0])
        pt_short = len(trades[(trades.value > 0) & (trades.direction == 'cover')])
        pt_long = len(trades[(trades.value > 0) & (trades.direction == 'sell')])
        total_trades = t_long + t_short

        print(self.symbol)
        print('Starting Equity: ' + str(trades['equity'][0]))
        print('Final Equity: ' + str(trades['equity'][-1]))
        print('Total return: {:.2%} '.format(float(total_ret)))
        print('Profitable trades: '+str(int(pt_count)))
        print('Losing trades: '+str(int(total_trades - pt_count)))

        if total_trades > 0:
            winners = pt_count / total_trades*100
            print('Winrate: ' + str(round(winners, 2)) + '%')
        else:
            print('No trades made')

        if t_short > 0:
            print('Short winrate: ' + str(round(pt_short/t_short*100, 2)) + '%')
        if t_long > 0:
            print('Long winrate: ' + str(round(pt_long/t_long*100, 2)) + '%')

        temp = trades.equity.resample('1D').last().dropna()
        temp.name = self.symbol
        idx = self.ohlc.close.resample('1D').last().index
        equity = pd.DataFrame(temp, index=idx).fillna(method='ffill').fillna(method='bfill')
        daily_ret = equity[self.symbol].pct_change()

        self.stats['symbol'] = self.symbol
        self.stats['daily_return'] = round(daily_ret.mean(), 4)
        self.stats['daily_risk'] = round(daily_ret.std(), 4)
        daily_excess = daily_ret - rfr/252
        sharpe = 252**0.5 * daily_excess.mean() / daily_excess.std()
        self.stats['sharpe'] = round(sharpe, 3)
        sortino = 252**0.5 * daily_excess.mean() / daily_excess[daily_excess < 0].std()
        self.stats['sortino'] = round(sortino, 3)
        period = (self.ohlc.index[-1] - self.ohlc.index[0]).total_seconds() / (31557600)
        cagr = (trades.equity.values[-1] / trades.equity.values[0]) ** (1/float(period))-1
        self.stats['cagr'] = round(cagr, 3)
        tval = trades.value.values
        pfr = tval[tval > 0].sum() / max(abs(tval[tval < 0].sum()), 1)
        self.stats['pfr'] = round(pfr, 3)
        maxdd = _max_draw(trades.equity.sort_index())
        closs = _consecutive_loss(trades.equity.sort_index())
        self.stats['maxdd'] = round(maxdd, 3)

        print('Daily return: {:.2%} '.format(float(daily_ret.mean())))
        print('Risk: {:.2%} '.format(float(daily_ret.std())))
        print('Sharpe: {:.2f} '.format(float(sharpe)))
        print('Sortino: {:.2f} '.format(float(sortino)))
        print('CAGR: {:.2%} '.format(float(cagr)))
        print('Profit factor: {:.2f}'.format(pfr))
        print('Consecutive losses: ' + str(closs))
        print('Max drawdown: {:.2%} '.format(float(maxdd)))
        print('\n')

        if plot:
            new_equity = trades.equity[(trades.equity != trades.equity.shift(1))]
            rolling_dd = new_equity.rolling(min_periods=1, window=2,
                                            center=False).apply(func=_max_rolling_dd)

            zipp = list(zip(new_equity, rolling_dd))
            df1 = pd.DataFrame(zipp, index=new_equity.index)
            df1.columns = ['Equity', 'Drawdown']

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            df1.plot(ax=ax1)
            ax1.set_ylabel('Portfolio value (USD)')
            ax1.set_xlabel('')
            plt.gcf().set_size_inches(12, 15)
            plt.show()
            plt.close()

    def plot_trades(self, startdate, enddate):
        ''' plot trades '''
        tlen = len(self.trades)
        if tlen > 0:
            trd = self.trades
            subset = slice(str(startdate), str(enddate))
            frm = trd.ix[subset]

            lent = frm.price[(frm.direction == 'buy') & (frm.lotsize > 0)]
            sent = frm.price[(frm.direction == 'short') & (frm.lotsize < 0)]
            lex = frm.price[(frm.direction == 'sell') & (frm.lotsize < 0)]
            sex = frm.price[(frm.direction == 'cover') & (frm.lotsize > 0)]

            fig = plt.figure()
            ax1 = fig.add_subplot(212)
            plt.gcf().set_size_inches(12, 15)
            ax1.set_ylabel('')
            ax1.set_xlabel('')

            if len(lent) > 0:
                pylab.plot(lent.index, lent.values, '^', color='lime', markersize=12,
                           label='long enter')
            if len(sent) > 0:
                pylab.plot(sent.index, sent.values, 'v', color='red', markersize=12,
                           label='short enter')
            if len(lex) > 0:
                pylab.plot(lex.index, lex.values, 'o', color='lime', markersize=7,
                           label='long exit')
            if len(sex) > 0:
                pylab.plot(sex.index, sex.values, 'o', color='red', markersize=7,
                           label='short exit')

            temp = pd.DataFrame(self.ohlc.open.ix[subset])
            temp.columns = [self.symbol+' Trades']
            temp.plot(ax=ax1, color='black')
            plt.show()
            plt.close()

            fig = plt.figure()
            ax1 = fig.add_subplot(212)
            plt.gcf().set_size_inches(12, 15)
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            eqt = pd.DataFrame(self.trades.ticks[subset].cumsum()*self.tick_size)
            eqt.columns = ['value']
            idx = eqt.index
            (eqt + self.ohlc.open[idx[0]]).plot(ax=ax1, color='red', style='-', label='value')
            self.ohlc.close.ix[subset].plot(ax=ax1, color='black', label='price')
            plt.show()
            plt.close()
        else:
            print('No trades to plot!')


    def annual_gains(self, start, end):
        ''' calculate annual gains '''
        gains = []
        years = []

        temp = self.trades.equity.resample('1D').last().dropna()
        temp.name = 'equity'
        idx = self.ohlc.close.resample('1D').last().index
        equity = pd.DataFrame(temp, index=idx).fillna(method='ffill').fillna(method='bfill')

        if self.trades.lotsize.abs().mean() == 1:
            for i in range(start, end+1, 1):
                gain = (equity[str(i)].equity[-1] - \
                                    equity[str(i)].equity[0]) / equity.equity[0]
                gains.append(gain*100)
                years.append(i)
        else:
            for i in range(start, end+1, 1):
                gain = (equity[str(i)].equity[-1] - \
                                    equity[str(i)].equity[0])/equity[str(i)].equity[0]
                gains.append(gain*100)
                years.append(i)

        zipp = list(zip(gains, years))
        df1 = pd.DataFrame(zipp)
        df1.columns = ['gains', 'years']

        fig = plt.figure()
        ax1 = fig.add_subplot(212)
        df1.plot(x='years', y='gains', ax=ax1, kind='bar', color='green')
        ax1.set_ylabel('Annual Returns (%)')
        ax1.set_xlabel('')
        plt.gcf().set_size_inches(12, 15)
        plt.show()
        plt.close()

    def analyze_results_ffn(self, rfr):
        ''' analyze performance with ffn'''
        data = self.trades.equity.resample('1D').last().dropna()
        data.name = self.symbol
        idx = self.ohlc.close.resample('1D').last().index
        equity = pd.DataFrame(data, index=idx).fillna(method='ffill').fillna(method='bfill')
        myffn = PerformanceStats(equity[self.symbol], rfr)
        print('\n')
        myffn.display()
        print('\n')
        myffn.display_monthly_returns()
        print('\n')
        #print(myffn.stats)
