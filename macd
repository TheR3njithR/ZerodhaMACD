from kiteconnect import KiteConnect
from math import floor, ceil
import datetime
import pandas as pd
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
# Contents taken from https://github.com/arkochhar/Technical-Indicators
# Contents taken from https://github.com/jigneshpylab/ZerodhaPythonScripts
api_key = "**"
api_secret = "**"
access_token = "access_token"
kite = KiteConnect(api_key=api_key)
print("[*] Generate Access Token :", kite.login_url())
request_token = input("[*] Enter Your Request Token here: ")
data = kite.generate_session(request_token, api_secret=api_secret)
kite.set_access_token(data["access_token"])

tickerlist = ["ZEEL", "YESBANK"]
tokenlist = [975873, 3050241]
NSELTPformate = ['NSE:{}'.format(i) for i in tickerlist]

def gethistoricaldata(token):
    enddate = datetime.datetime.today()
    startdate = enddate - datetime.timedelta(10)
    df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    try:
        data = kite.historical_data(token, startdate, enddate, interval=candlesize)
        df = pd.DataFrame.from_dict(data, orient='columns', dtype=None)
        if not df.empty:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = df['date'].astype(str).str[:-6]
            df['date'] = pd.to_datetime(df['date'])
            macd(df)
    except Exception as e:
        print("         error in gethistoricaldata", token, e)
    return df

orderslist = []

def macd(df, fastEMA=12, slowEMA=26, signal=9, base='Close'):
    """
    Function to compute Moving Average Convergence Divergence (MACD)

    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        fastEMA : Integer indicates faster EMA
        slowEMA : Integer indicates slower EMA
        signal : Integer indicates the signal generator for MACD
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)

    Returns :
        df : Pandas DataFrame with new columns added for
            Fast EMA (ema_$fastEMA)
            Slow EMA (ema_$slowEMA)
            MACD (macd_$fastEMA_$slowEMA_$signal)
            MACD Signal (signal_$fastEMA_$slowEMA_$signal)
            MACD Histogram (MACD (hist_$fastEMA_$slowEMA_$signal))
    """
    try:
        fE = "ema_" + str(fastEMA)
        sE = "ema_" + str(slowEMA)
        macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
        hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

        # Compute fast and slow EMA
        ema(df, base, fE, fastEMA)
        ema(df, base, sE, slowEMA)

        # ema12 = ema(df, base, fE, fastEMA)
        # ema26 = ema(df, base, sE, slowEMA)

        exp3 = ema(df, base, sE, slowEMA) - ema(df, base, fE, fastEMA)

        # Compute MACD
        # df[macd] = np.where(np.logical_and(np.logical_not(df[fE] == 0), np.logical_not(df[sE] == 0)), df[fE] - df[sE], 0)

        # Compute MACD Signal
        ema(df, macd, sig, signal)

        plt.plot(df, macd, label='MACD', color = '#EBD2BE')
        plt.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')

        # Compute MACD Histogram
        df[hist] = np.where(np.logical_and(np.logical_not(df[macd] == 0), np.logical_not(df[sig] == 0)), df[macd] - df[sig],
                            0)

        return df
    except Exception as e:
        print(e)
        def run_strategy():
    for i in range(0, len(tickerlist)):
        if tickerlist[i] in orderslist:
            continue
        try:
            gethistoricaldata(tokenlist[i])


            # histdata = gethistoricaldata(tokenlist[i])
            # histdata["FMA"] = histdata['close'].rolling(fastMA_period).mean()
            # histdata["SMA"] = histdata['close'].rolling(slowMA_period).mean()
            # FMA = histdata.FMA.values[-10:]
            # SMA = histdata.SMA.values[-10:]
            # MA_diff = SMA - FMA
            #
            # lastclose = histdata.close.values[-1]
            # stoploss_buy = histdata.low.values[-3]  # third last candle as stoploss
            # stoploss_sell = histdata.high.values[-3]  # third last candle as stoploss
            #
            # if stoploss_buy > lastclose * 0.996:
            #     stoploss_buy = lastclose * 0.996  # minimum stoploss as 0.4 %
            #
            # if stoploss_sell < lastclose * 1.004:
            #     stoploss_sell = lastclose * 1.004  # minimum stoploss as 0.4 %
            # print(tickerlist[i], lastclose, " FMA", FMA[-1], " SMA", SMA[-1], " MA_diff", MA_diff[-1])
            #
            # if MA_diff[-1] > 0:  # and  MA_diff[-3]< 0 :
            #     stoploss_buy = lastclose - stoploss_buy
            #     # quantity = floor(max(1, (risk_per_trade / stoploss_buy)))
            #     quantity = 2
            #     target = stoploss_buy * 3  # risk reward as 3
            #     price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
            #     stoploss_buy = int(100 * (floor(stoploss_buy / 0.05) * 0.05)) / 100
            #     quantity = int(quantity)
            #     target = int(100 * (floor(target / 0.05) * 0.05)) / 100
            #     orderslist.append(tickerlist[i])
            #     order = kite.place_order(exchange='NSE',
            #                              tradingsymbol=tickerlist[i],
            #                              transaction_type="BUY",
            #                              quantity=quantity,
            #                              price=price,
            #                              product='MIS',
            #                              order_type='LIMIT',
            #                              validity='DAY',
            #                              trigger_price='0',
            #                              squareoff=target,
            #                              stoploss=stoploss_buy,
            #                              # trailing_stoploss=trailing_loss,
            #                              variety="bo"
            #                              )
            #     print("         Order : ", "BUY", tickerlist[i], "quantity:", quantity, "target:", target, "stoploss:",
            #           stoploss_buy, datetime.datetime.now())
            #
            # if MA_diff[-1] < 0 and MA_diff[-3] > 0:
            #     stoploss_sell = stoploss_sell - lastclose
            #     # quantity = floor(max(1, (risk_per_trade / stoploss_sell)))
            #     quantity = 2
            #     target = stoploss_sell * 3  # risk reward as 3
            #     price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
            #     stoploss_sell = int(100 * (floor(stoploss_sell / 0.05) * 0.05)) / 100
            #     quantity = int(quantity)
            #     target = int(100 * (floor(target / 0.05) * 0.05)) / 100
            #     orderslist.append(tickerlist[i])
            #     # order = kite.place_order(exchange='NSE',
            #     #                          tradingsymbol=tickerlist[i],
            #     #                          transaction_type="SELL",
            #     #                          quantity=quantity,
            #     #                          price=price,
            #     #                          product='MIS',
            #     #                          order_type='LIMIT',
            #     #                          validity='DAY',
            #     #                          trigger_price='0',
            #     #                          squareoff=target,
            #     #                          stoploss=stoploss_sell,
            #     #                          # trailing_stoploss=trailing_loss,
            #     #                          variety="bo"
            #     #                          )
            #     print("         Order : ", "SELL", tickerlist[i], "quantity:", quantity, "target:", target, "stoploss:",
            #           stoploss_sell, datetime.datetime.now())
        except Exception as e:
            print(e)
        # print("orderslist", orderslist)


def run():
    global runcount
    start_time = int(9) * 60 + int(33)  # specify in int (hr) and int (min) foramte
    end_time = int(15) * 60 + int(10)  # do not place fresh order
    stop_time = int(15) * 60 + int(15)  # square off all open positions
    last_time = start_time
    schedule_interval = 180  # run at every 3 min

    while True:
        if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= end_time:
            if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= stop_time:
                print(sys._getframe().f_lineno, "Trading day closed, time is above stop_time")
                break

        elif (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= start_time:
            if time.time() >= last_time:
                last_time = time.time() + schedule_interval
                print("\n\n {} Run Count : Time - {} ".format(runcount, datetime.datetime.now()))
                if runcount >= 0:
                    try:
                        run_strategy()
                    except Exception as e:
                        print("Run error", e)
                runcount = runcount + 1
        else:
            print('     Waiting...', datetime.datetime.now())
            time.sleep(1)


def ema(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """
    try:
        con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

        if (alpha == True):
            # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
            df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
        else:
            # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
            df[target] = con.ewm(span=period, adjust=False).mean()

        df[target].fillna(0, inplace=True)
        return df

    except Exception as e:
        print(e)

runcount = 0
run()
