import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
import random
from ta.volatility import BollingerBands
from tapy import Indicators
from scipy import stats
import os

class StockDataDownloader:
    def __init__(self, ticker, period, interval):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df = None

    def download_data(self):
        self.df = yf.download(tickers=self.ticker, period=self.period, interval=self.interval)
        return self.df

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def remove_zero_volume(self):
        self.df = self.df[self.df['Volume'] != 0]
        return self.df

    def add_time_columns(self):
        self.df.index = pd.to_datetime(self.df.index)
        self.df['Year'] = self.df.index.year
        self.df['Month'] = self.df.index.month
        self.df['Day'] = self.df.index.day
        self.df['Hour'] = self.df.index.hour
        self.df['Minute'] = self.df.index.minute
        self.df.reset_index(drop=True, inplace=True)
        return self.df

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df

    def calculate_rsi(self):
        self.df['RSI'] = ta.rsi(self.df['Close'], length=14).fillna(method='bfill')
        return self.df

    def calculate_ema(self):
        self.df['EMA'] = ta.ema(self.df['Close'], length=4).fillna(method='bfill')
        return self.df

class SignalGenerator2:
    def __init__(self, df, backcandles=15):
        self.df = df
        self.backcandles = backcandles
        self.EMAsignal = [0] * len(df)

    def generate_ema_signal(self):
        for row in range(self.backcandles, len(self.df)):
            upt = 1
            dnt = 1
            for i in range(row - self.backcandles, row + 1):
                if max(self.df.Open[i], self.df.Close[i]) >= self.df.EMA[i]:
                    dnt = 0
                if min(self.df.Open[i], self.df.Close[i]) <= self.df.EMA[i]:
                    upt = 0
            if upt == 1 and dnt == 1:
                self.EMAsignal[row] = 3
            elif upt == 1:
                self.EMAsignal[row] = 2
            elif dnt == 1:
                self.EMAsignal[row] = 1
        self.df['EMAsignal'] = self.EMAsignal
        return self.df
    
class PivotSignalGenerator:
    def __init__(self, df, window=5):
        self.df = df
        self.window = window

    def is_pivot(self, candle):
        if candle - self.window < 0 or candle + self.window >= len(self.df):
            return 0

        pivotHigh = 1
        pivotLow = 2
        for i in range(candle - self.window, candle + self.window + 1):
            if self.df.iloc[candle].Low > self.df.iloc[i].Low:
                pivotLow = 0
            if self.df.iloc[candle].High < self.df.iloc[i].High:
                pivotHigh = 0
        if pivotHigh and pivotLow:
            return 3
        elif pivotHigh:
            return pivotHigh
        elif pivotLow:
            return pivotLow
        else:
            return 0

    def pointpos(self, row):
        if row['isPivot'] == 2:
            return row['Low'] - 1e-3
        elif row['isPivot'] == 1:
            return row['High'] + 1e-3
        else:
            return np.nan

    def generate_pivot_signals(self):
        self.df['isPivot'] = self.df.apply(lambda x: self.is_pivot(x.name), axis=1)
        self.df['pointpos'] = self.df.apply(lambda row: self.pointpos(row), axis=1)
        return self.df

class CHOCHPatternDetector:
    def __init__(self, df, backcandles=40, window=6):
        self.df = df
        self.backcandles = backcandles
        self.window = window

    def detect_structure(self, candle):
        localdf = self.df[candle-self.backcandles-self.window:candle-self.window]
        highs = localdf[localdf['isPivot'] == 1].High.tail(3).values
        idxhighs = localdf[localdf['isPivot'] == 1].High.tail(3).index
        lows = localdf[localdf['isPivot'] == 2].Low.tail(3).values
        idxlows = localdf[localdf['isPivot'] == 2].Low.tail(3).index

        pattern_detected = False

        lim1 = self.df['High'].mean() - self.df['Low'].mean()
        lim2 = lim1 / 3
        if len(highs) == 3 and len(lows) == 3:
            order_condition = (idxlows[0] < idxhighs[0] < idxlows[1] < idxhighs[1] < idxlows[2] < idxhighs[2])
            diff_condition = (
                abs(lows[0] - highs[0]) > lim1 and
                abs(highs[0] - lows[1]) > lim2 and
                abs(highs[1] - lows[1]) > lim1 and
                abs(highs[1] - lows[2]) > lim2
            )
            pattern_1 = (
                lows[0] < highs[0] and
                lows[1] > lows[0] and lows[1] < highs[0] and
                highs[1] > highs[0] and
                lows[2] > lows[1] and lows[2] < highs[1] and
                highs[2] < highs[1] and highs[2] > lows[2]
            )

            pattern_2 = (
                lows[0] < highs[0] and
                lows[1] > lows[0] and lows[1] < highs[0] and
                highs[1] > highs[0] and
                lows[2] < lows[1] and
                highs[2] < highs[1]
            )

            if order_condition and diff_condition and (pattern_1 or pattern_2):
                pattern_detected = True

        return 1 if pattern_detected else 0

    def generate_choch_pattern(self):
        self.df['CHOCH_pattern_detected'] = self.df.index.map(lambda x: self.detect_structure(x))
        return self.df

class FibonacciSignalGenerator:
    def __init__(self, df, backcandles=15, gap_candles=4):
        self.df = df
        self.backcandles = backcandles
        self.gap_candles = gap_candles
        self.zone_threshold = df['High'].mean() - df['Low'].mean()
        self.price_diff_threshold = df['High'].mean() - df['Low'].mean()

    def generate_signal(self, l):
        if l < self.backcandles + self.gap_candles or l >= len(self.df):
            return (0, 0, 0, 0, 0)  # Return neutral signal if not enough data

        max_price = self.df.High[l - self.backcandles:l - self.gap_candles].max()
        min_price = self.df.Low[l - self.backcandles:l - self.gap_candles].min()
        index_max = self.df.High[l - self.backcandles:l - self.gap_candles].idxmax()
        index_min = self.df.Low[l - self.backcandles:l - self.gap_candles].idxmin()
        price_diff = max_price - min_price

        if (self.df.EMAsignal[l] == 2 and
            (index_min < index_max) and
            price_diff > self.price_diff_threshold):

            entry_price = max_price - 0.52 * price_diff
            stop_loss = max_price - 0.68 * price_diff
            take_profit = max_price - 0.0 * price_diff

            if abs(self.df.Close[l] - entry_price) < self.zone_threshold and self.df.High[l - self.gap_candles:l].min() > entry_price:
                return (2, stop_loss, take_profit, index_min, index_max)
            else:
                return (0, 0, 0, 0, 0)

        elif (self.df.EMAsignal[l] == 1 and
              (index_min > index_max) and
              price_diff > self.price_diff_threshold):

            entry_price = min_price + 0.52 * price_diff
            stop_loss = min_price + 0.68 * price_diff
            take_profit = min_price + 0.0 * price_diff

            if abs(self.df.Close[l] - entry_price) < self.zone_threshold and self.df.Low[l - self.gap_candles:l].max() < entry_price:
                return (1, stop_loss, take_profit, index_min, index_max)
            else:
                return (0, 0, 0, 0, 0)

        else:
            return (0, 0, 0, 0, 0)

    def generate_fibonacci_signals(self):
        signal = [0] * len(self.df)
        TP = [0] * len(self.df)
        SL = [0] * len(self.df)
        MinSwing = [0] * len(self.df)
        MaxSwing = [0] * len(self.df)

        for row in range(self.backcandles, len(self.df)):
            gen_sig = self.generate_signal(row)
            signal[row] = gen_sig[0]
            SL[row] = gen_sig[1]
            TP[row] = gen_sig[2]
            MinSwing[row] = gen_sig[3]
            MaxSwing[row] = gen_sig[4]
        
        self.df['fibonacci_signal'] = signal
        self.df['SL'] = SL
        self.df['TP'] = TP
        self.df['MinSwing'] = MinSwing
        self.df['MaxSwing'] = MaxSwing
        
        return self.df
class LevelBreakSignalDetector:
    def __init__(self, df, lbd_backcandles=40, lbd_window=6, lbh_backcandles=60, lbh_window=11):
        self.df = df
        self.lbd_backcandles = lbd_backcandles
        self.lbd_window = lbd_window
        self.lbh_backcandles = lbh_backcandles
        self.lbh_window = lbh_window
        self.zone_width = df['High'].mean() - df['Low'].mean()

    def detect_structure(self, candle, backcandles, window):
        if (candle <= (backcandles + window)) or (candle + window + 1 >= len(self.df)):
            return 0

        localdf = self.df.iloc[candle - backcandles - window : candle - window]
        highs = localdf[localdf['isPivot'] == 1].High.tail(3).values
        lows = localdf[localdf['isPivot'] == 2].Low.tail(3).values
        levelbreak = 0

        if len(lows) == 3:
            support_condition = True
            mean_low = lows.mean()
            for low in lows:
                if abs(low - mean_low) > self.zone_width:
                    support_condition = False
                    break
            if support_condition and (mean_low - self.df.loc[candle].Close) > self.zone_width * 2:
                levelbreak = 1

        if len(highs) == 3:
            resistance_condition = True
            mean_high = highs.mean()
            for high in highs:
                if abs(high - mean_high) > self.zone_width:
                    resistance_condition = False
                    break
            if resistance_condition and (self.df.loc[candle].Close - mean_high) > self.zone_width * 2:
                levelbreak = 2

        return levelbreak

    def generate_lbd_signals(self):
        self.df['LBD_detected'] = self.df.apply(lambda row: self.detect_structure(row.name, self.lbd_backcandles, self.lbd_window), axis=1)
        return self.df

    def generate_lbh_signals(self):
        self.df['LBH_detected'] = self.df.apply(lambda row: self.detect_structure(row.name, self.lbh_backcandles, self.lbh_window), axis=1)
        return self.df

    def generate_level_break_signals(self):
        self.generate_lbd_signals()
        self.generate_lbh_signals()
        return self.df

class SupportResistanceSignalDetector:
    def __init__(self, df, n1=8, n2=6, backcandles=140):
        self.df = df
        self.n1 = n1
        self.n2 = n2
        self.backcandles = backcandles
        self.wick_threshold = df['High'].max() - df['Low'].min()

    def support(self, df1, l, n1, n2):
        if df1.Low[l-n1:l].min() < df1.Low[l] or df1.Low[l+1:l+n2+1].min() < df1.Low[l]:
            return 0

        candle_body = abs(df1.Open[l] - df1.Close[l])
        lower_wick = min(df1.Open[l], df1.Close[l]) - df1.Low[l]
        if lower_wick > candle_body and lower_wick > self.wick_threshold:
            return 1

        return 0

    def resistance(self, df1, l, n1, n2):
        if df1.High[l-n1:l].max() > df1.High[l] or df1.High[l+1:l+n2+1].max() > df1.High[l]:
            return 0

        candle_body = abs(df1.Open[l] - df1.Close[l])
        upper_wick = df1.High[l] - max(df1.Open[l], df1.Close[l])
        if upper_wick > candle_body and upper_wick > self.wick_threshold:
            return 1

        return 0

    def close_resistance(self, l, levels, lim, df):
        if len(levels) == 0:
            return 0
        c1 = abs(df.High[l] - min(levels, key=lambda x: abs(x - df.High[l]))) <= lim
        c2 = abs(max(df.Open[l], df.Close[l]) - min(levels, key=lambda x: abs(x - df.High[l]))) <= lim
        c3 = min(df.Open[l], df.Close[l]) < min(levels, key=lambda x: abs(x - df.High[l]))
        c4 = df.Low[l] < min(levels, key=lambda x: abs(x - df.High[l]))
        if (c1 or c2) and c3 and c4:
            return min(levels, key=lambda x: abs(x - df.High[l]))
        else:
            return 0

    def close_support(self, l, levels, lim, df):
        if len(levels) == 0:
            return 0
        c1 = abs(df.Low[l] - min(levels, key=lambda x: abs(x - df.Low[l]))) <= lim
        c2 = abs(min(df.Open[l], df.Close[l]) - min(levels, key=lambda x: abs(x - df.Low[l]))) <= lim
        c3 = max(df.Open[l], df.Close[l]) > min(levels, key=lambda x: abs(x - df.Low[l]))
        c4 = df.High[l] > min(levels, key=lambda x: abs(x - df.Low[l]))
        if (c1 or c2) and c3 and c4:
            return min(levels, key=lambda x: abs(x - df.Low[l]))
        else:
            return 0

    def is_below_resistance(self, l, level_backCandles, level, df):
        return df.loc[l - level_backCandles : l - 1, 'High'].max() < level

    def is_above_support(self, l, level_backCandles, level, df):
        return df.loc[l - level_backCandles : l - 1, 'Low'].min() > level

    def check_candle_signal(self, l, n1, n2, backCandles, df):
        ss = []
        rr = []
        for subrow in range(l - backCandles, l - n2):
            if self.support(df, subrow, n1, n2):
                ss.append(df.Low[subrow])
            if self.resistance(df, subrow, n1, n2):
                rr.append(df.High[subrow])

        ss.sort()
        for i in range(1, len(ss)):
            if i >= len(ss):
                break
            if abs(ss[i] - ss[i-1]) <= 0.0001:
                ss.pop(i)

        rr.sort(reverse=True)
        for i in range(1, len(rr)):
            if i >= len(rr):
                break
            if abs(rr[i] - rr[i-1]) <= 0.0001:
                rr.pop(i)

        rrss = rr + ss
        rrss.sort()
        for i in range(1, len(rrss)):
            if i >= len(rrss):
                break
            if abs(rrss[i] - rrss[i-1]) <= 0.0001:
                rrss.pop(i)

        cR = self.close_resistance(l, rrss, 150e-5, df)
        cS = self.close_support(l, rrss, 150e-5, df)

        if cR and self.is_below_resistance(l, 6, cR, df) and df.RSI[l-1:l].min() < 45:
            return 1
        elif cS and self.is_above_support(l, 6, cS, df) and df.RSI[l-1:l].max() > 55:
            return 2
        else:
            return 0

    def generate_support_resistance_signals(self):
        signal = [0 for _ in range(len(self.df))]
        for row in tqdm(range(self.backcandles + self.n1, len(self.df) - self.n2)):
            signal[row] = self.check_candle_signal(row, self.n1, self.n2, self.backcandles, self.df)
        self.df["SR_signal"] = signal
        return self.df

class ChannelDetector:
    def __init__(self, df, backcandles, window):
        self.df = df
        self.backcandles = backcandles
        self.window = window

    def collect_channel(self, candle):
        highs = self.df[self.df['isPivot'] == 1].High.values
        idxhighs = self.df[self.df['isPivot'] == 1].High.index
        lows = self.df[self.df['isPivot'] == 2].Low.values
        idxlows = self.df[self.df['isPivot'] == 2].Low.index

        if len(lows) >= 3 and len(highs) >= 3:
            sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows, lows)
            sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs, highs)

            return (sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
        else:
            return (0, 0, 0, 0, 0, 0)

    def generate_channel_signals(self):
        self.df['Channel'] = [self.collect_channel(candle) for candle in self.df.index]
        return self.df
    
class BreakoutDetector:
    def __init__(self, df, backcandles, window):
        self.df = df
        self.backcandles = backcandles
        self.window = window

    def isBreakOut(self, candle):
        if (candle - self.backcandles - self.window) < 0:
            return 0

        sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = self.df.iloc[candle].Channel

        prev_idx = candle - 1
        prev_high = self.df.iloc[prev_idx].High
        prev_low = self.df.iloc[prev_idx].Low
        prev_close = self.df.iloc[prev_idx].Close

        curr_high = self.df.iloc[candle].High
        curr_low = self.df.iloc[candle].Low
        curr_close = self.df.iloc[candle].Close
        curr_open = self.df.iloc[candle].Open

        if (prev_high > (sl_lows * prev_idx + interc_lows) and
            prev_close < (sl_lows * prev_idx + interc_lows) and
            curr_open < (sl_lows * candle + interc_lows) and
            curr_close < (sl_lows * prev_idx + interc_lows)):
            return 1

        elif (prev_low < (sl_highs * prev_idx + interc_highs) and
              prev_close > (sl_highs * prev_idx + interc_highs) and
              curr_open > (sl_highs * candle + interc_highs) and
              curr_close > (sl_highs * prev_idx + interc_highs)):
            return 2

        else:
            return 0

    def detect_breakouts(self):
        self.df["isBreakOut"] = [self.isBreakOut(candle) for candle in self.df.index]
        return self.df

class CandlestickPatternDetector:
    def __init__(self, df):
        self.df = df
        self.length = len(df)
        self.High = list(df['High'])
        self.Low = list(df['Low'])
        self.Close = list(df['Close'])
        self.Open = list(df['Open'])
        self.bodydiff = [0] * self.length
        self.Highdiff = [0] * self.length
        self.Lowdiff = [0] * self.length
        self.ratio1 = [0] * self.length
        self.ratio2 = [0] * self.length

    def isEngulfing(self, l):
        row = l
        self.bodydiff[row] = abs(self.Open[row] - self.Close[row])
        if self.bodydiff[row] < 0.00001:
            self.bodydiff[row] = 0.00001

        bodydiffmin = 0.02
        if (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
            self.Open[row - 1] < self.Close[row - 1] and
            self.Open[row] > self.Close[row] and
            (self.Open[row] - self.Close[row - 1]) >= -0e-5 and self.Close[row] < self.Open[row - 1]):
            return 1

        elif (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
              self.Open[row - 1] > self.Close[row - 1] and
              self.Open[row] < self.Close[row] and
              (self.Open[row] - self.Close[row - 1]) <= +0e-5 and self.Close[row] > self.Open[row - 1]):
            return 2
        else:
            return 0

    def isEngulfingStrong(self, l):
        row = l
        self.bodydiff[row] = abs(self.Open[row] - self.Close[row])
        if self.bodydiff[row] < 0.00001:
            self.bodydiff[row] = 0.00001

        bodydiffmin = 0.02
        if (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
            self.Open[row - 1] < self.Close[row - 1] and
            self.Open[row] > self.Close[row] and
            (self.Open[row] - self.Close[row - 1]) >= -0e-5 and self.Close[row] < self.Low[row - 1]):
            return 1

        elif (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
              self.Open[row - 1] > self.Close[row - 1] and
              self.Open[row] < self.Close[row] and
              (self.Open[row] - self.Close[row - 1]) <= +0e-5 and self.Close[row] > self.High[row - 1]):
            return 2
        else:
            return 0

    def isStar(self, l):
        bodydiffmin = 0.020
        row = l
        self.Highdiff[row] = self.High[row] - max(self.Open[row], self.Close[row])
        self.Lowdiff[row] = min(self.Open[row], self.Close[row]) - self.Low[row]
        self.bodydiff[row] = abs(self.Open[row] - self.Close[row])
        if self.bodydiff[row] < 0.00001:
            self.bodydiff[row] = 0.00001
        self.ratio1[row] = self.Highdiff[row] / self.bodydiff[row]
        self.ratio2[row] = self.Lowdiff[row] / self.bodydiff[row]

        if (self.ratio1[row] > 1 and self.Lowdiff[row] < 0.2 * self.Highdiff[row] and self.bodydiff[row] > bodydiffmin):
            return 1
        elif (self.ratio2[row] > 1 and self.Highdiff[row] < 0.2 * self.Lowdiff[row] and self.bodydiff[row] > bodydiffmin):
            return 2
        else:
            return 0

    def direction(self, l):
        if (self.Open[l] > self.Close[l]):
            return 1
        elif (self.Open[l] < self.Close[l]):
            return 2
        else:
            return 0

    def Revsignal1(self):
        signal = [0] * self.length
        for row in range(1, self.length):
            if ((self.isEngulfing(row) == 1 and self.isStar(row) == 1)):
                signal[row] = 1
            elif ((self.isEngulfing(row) == 2 and self.isStar(row) == 2)):
                signal[row] = 2
            else:
                signal[row] = 0
        return signal

    def detect_candlestick_patterns(self):
        self.df['candlestick_signal'] = self.Revsignal1()
        return self.df

import pandas as pd

class TrendTargeting:
    def __init__(self, df, barsfront):
        self.df = df
        self.barsfront = barsfront
        self.piplim = 200e-5
        self.trendcat = [None] * len(df)

    def analyze_trends(self):
        length = len(self.df)
        High = list(self.df['High'])
        Low = list(self.df['Low'])
        Close = list(self.df['Close'])
        Open = list(self.df['Open'])

        for line in range(0, length - 1 - self.barsfront):
            for i in range(1, self.barsfront + 1):
                if ((High[line + i] - Close[line]) > self.piplim) and ((Close[line] - Low[line + i]) > self.piplim):
                    self.trendcat[line] = 3  # no trend
                    break
                elif (Close[line] - Low[line + i]) > self.piplim:
                    self.trendcat[line] = 1  #-1 downtrend
                    break
                elif (High[line + i] - Close[line]) > self.piplim:
                    self.trendcat[line] = 2  # uptrend
                    break
                else:
                    self.trendcat[line] = 0  # no clear trend

    def get_trend_categories(self):
        self.analyze_trends()
        self.df['Trend'] = self.trendcat
        return self.df

class TrendResultant:
    def __init__(self, df):
        self.df = df

    def apply_conditions(self):
        conditions = [
            ((self.df['Trend'] == 1) | (self.df['Trend'] == 3)) & (self.df['candlestick_signal'] == 1),
            ((self.df['Trend'] == 2) | (self.df['Trend'] == 3)) & (self.df['candlestick_signal'] == 2)
        ]
        values = [1, 2]
        self.df['result'] = np.select(conditions, values)
        return self.df

    def calculate_ratio(self, trendId):
        return self.df[self.df['result'] == trendId].result.count() / self.df[self.df['candlestick_signal'] == trendId].candlestick_signal.count()

    def filter_trends(self, trendId):
        filtered_df = self.df[(self.df['Trend'] != trendId) & (self.df['Trend'] != 3) & (self.df['candlestick_signal'] == trendId)]
        return filtered_df  

import pandas as pd

class EngulfingPatternDetector:
    def __init__(self, df):
        self.df = df
        self.length = len(df)
        self.signal = [0] * self.length
        self.bodydiff = [0] * self.length
        self.detect_signals()
        self.df['signal1'] = self.signal

    def detect_signals(self):
        High = list(self.df['High'])
        Low = list(self.df['Low'])
        Close = list(self.df['Close'])
        Open = list(self.df['Open'])
        for row in range(1, self.length):
            self.bodydiff[row] = abs(Open[row] - Close[row])
            bodydiffmin = 0.003
            if (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
                    Open[row - 1] < Close[row - 1] and
                    Open[row] > Close[row] and
                    (Open[row] - Close[row - 1]) >= +0e-5 and Close[row] < Open[row - 1]):
                self.signal[row] = 1
            elif (self.bodydiff[row] > bodydiffmin and self.bodydiff[row - 1] > bodydiffmin and
                  Open[row - 1] > Close[row - 1] and
                  Open[row] < Close[row] and
                  (Open[row] - Close[row - 1]) <= -0e-5 and Close[row] > Open[row - 1]):
                self.signal[row] = 2
            else:
                self.signal[row] = 0

    def get_signal_counts(self):
        return self.df[self.df['signal1'] == 1].count()

class BollingerIndicators:
    def __init__(self, df):
        self.df = df
    
    def calculate_moving_average(self, window=20):
        self.df['ma20'] = self.df['Close'].rolling(window=window).mean()
        self.df['ma20'].fillna(self.df['ma20'].bfill(), inplace=True)
    
    def calculate_bollinger_bands(self, window=20, window_dev=2):
        indicator_bb = BollingerBands(close=self.df["Close"], window=window, window_dev=window_dev)
        self.df['middle_band'] = indicator_bb.bollinger_mavg()
        self.df['upper_band'] = indicator_bb.bollinger_hband()
        self.df['lower_band'] = indicator_bb.bollinger_lband()
        self.df['middle_band'].fillna(self.df['middle_band'].bfill(), inplace=True)
        self.df['upper_band'].fillna(self.df['upper_band'].bfill(), inplace=True)
        self.df['lower_band'].fillna(self.df['lower_band'].bfill(), inplace=True)

class BollingerBandStrategy:
    def __init__(self, df):
        self.df = df
 
    def calculate_signals(self):
        self.df['buy_signal'] = self.df['Close'] > self.df['upper_band']
        self.df['sell_signal'] = self.df['Close'] < self.df['ma20']

        # Initialize position column
        self.df['Position'] = 0

        # Buy when the price breaks above the upper Bollinger Band
        self.df.loc[self.df['buy_signal'], 'Position'] = 1

        # Sell when the price drops below the 20-day moving average
        self.df.loc[self.df['sell_signal'], 'Position'] = -1

        # Forward fill the position column to ensure we hold onto our position
        # until a sell signal is generated
        self.df['Position'].fillna(method='ffill', inplace=True)

        # Calculate the daily returns of the strategy
        self.df['Strategy Returns'] = self.df['Close'].pct_change() * self.df['Position'].shift(1)

        return self.df

import tapy as tap
from tapy import Indicators
import pandas as pd

class FractalStrategy:
    def __init__(self, df):
        self.df = df
        self.calculate_fractal()
        self.calculate_fractals()
        self.generate_signals()
    
    def calculate_fractal(self):
        self.df['fractal_high'] = self.df['High'][(self.df['High'].shift(2) < self.df['High'].shift(1)) & 
                                                  (self.df['High'].shift(1) < self.df['High']) & 
                                                  (self.df['High'] > self.df['High'].shift(-1)) & 
                                                  (self.df['High'] > self.df['High'].shift(-2))]
        
        self.df['fractal_low'] = self.df['Low'][(self.df['Low'].shift(2) > self.df['Low'].shift(1)) & 
                                                (self.df['Low'].shift(1) > self.df['Low']) & 
                                                (self.df['Low'] < self.df['Low'].shift(-1)) & 
                                                (self.df['Low'] < self.df['Low'].shift(-2))]

        self.df['fractal_high'].fillna(method='ffill', inplace=True)
        self.df['fractal_low'].fillna(method='ffill', inplace=True)

    def calculate_fractals(self):
        self.df['fractals_high'] = self.df['High'][(self.df['High'].shift(3) < self.df['High'].shift(2)) & 
                                                  (self.df['High'].shift(2) < self.df['High'].shift(1)) & 
                                                  (self.df['High'].shift(1) < self.df['High']) & 
                                                  (self.df['High'] > self.df['High'].shift(-1)) & 
                                                  (self.df['High'] > self.df['High'].shift(-2)) & 
                                                  (self.df['High'] > self.df['High'].shift(-3))]
        
        self.df['fractals_low'] = self.df['Low'][(self.df['Low'].shift(3) > self.df['Low'].shift(2)) & 
                                                (self.df['Low'].shift(2) > self.df['Low'].shift(1)) & 
                                                (self.df['Low'].shift(1) > self.df['Low']) & 
                                                (self.df['Low'] < self.df['Low'].shift(-1)) & 
                                                (self.df['Low'] < self.df['Low'].shift(-2)) & 
                                                (self.df['Low'] < self.df['Low'].shift(-3))]

        self.df['fractals_high'].fillna(method='ffill', inplace=True)
        self.df['fractals_low'].fillna(method='ffill', inplace=True)
    
    def generate_signals(self):
        # Buy when the price breaks above the fractal high
        self.df['buy_signal1'] = self.df['Close'] > self.df['fractal_high']
        self.df.loc[self.df['buy_signal1'] == True, 'Position'] = 1

        # Sell when the price drops below the fractal low
        self.df['sell_signal1'] = self.df['Close'] < self.df['fractal_low']
        self.df.loc[self.df['sell_signal1'] == True, 'Position'] = -1

        # Fill missing positions
        self.df['Position'].fillna(method='ffill', inplace=True)
        
        # Calculate the daily returns of the strategy
        self.df['Strategy Returns'] = self.df['Close'].pct_change() * self.df['Position'].shift(1)

    def update_fractals(self):
        self.calculate_fractals()

    def get_df(self):
        return self.df

class VSignalGenerator:
    def __init__(self, df, vbackcandles=1):
        self.df = df
        self.vbackcandles = vbackcandles

    def generate_vsignal(self):
        VSignal = [0] * len(self.df)
        for row in range(self.vbackcandles + 1, len(self.df)):
            VSignal[row] = 1
            for i in range(row - self.vbackcandles, row):
                if self.df.Volume[row] < self.df.Volume[i] and self.df.Volume[row - 1] < self.df.Volume[row - 2]:
                    VSignal[row] = 0
        return VSignal

    def add_vsignal_column(self):
        self.df['VSignal'] = self.generate_vsignal()

class PriceSignalGenerator:
    def __init__(self, df, pbackcandles):
        self.df = df
        self.pbackcandles = pbackcandles
        self.PriceSignal = [0] * len(self.df)

    def generate_price_signal(self):
        for row in range(self.pbackcandles, len(self.df)):
            self.PriceSignal[row] = 1
            for i in range(row - self.pbackcandles, row):
                if self.df.EMAsignal[row] == 1:  # downtrend
                    if self.df.Open[row] <= self.df.Close[row]:  # downcandle row
                        self.PriceSignal[row] = 0
                    elif self.df.Open[i] > self.df.Close[i]:  # downcandle i we are looking for 4 upcandles
                        self.PriceSignal[row] = 0
                elif self.df.EMAsignal[row] == 2:  # uptrend
                    if self.df.Open[row] >= self.df.Close[row]:  # upcandle row
                        self.PriceSignal[row] = 0
                    elif self.df.Open[i] < self.df.Close[i]:  # upcandle i we are looking for 4 dowcandles
                        self.PriceSignal[row] = 0
                else:
                    self.PriceSignal[row] = 0

        self.df['PriceSignal'] = self.PriceSignal
        return self.df['PriceSignal']

class SignalProcessor:
    def __init__(self, df):
        self.df = df

    def calculate_tot_signal(self):
        TotSignal = [0] * len(self.df)
        for row in range(len(self.df)):
            if self.df.EMAsignal[row] == 1 and self.df.VSignal[row] == 1 and self.df.PriceSignal[row] == 1:
                TotSignal[row] = 1
            elif self.df.EMAsignal[row] == 2 and self.df.VSignal[row] == 1 and self.df.PriceSignal[row] == 1:
                TotSignal[row] = 2
        self.df['TotSignal'] = TotSignal

    def get_tot_signal(self):
        if 'TotSignal' not in self.df.columns:
            self.calculate_tot_signal()
        return self.df['TotSignal']
    
class SLSignalCalculator:
    def __init__(self, df, SLbackcandles=4):
        self.df = df
        self.SLbackcandles = SLbackcandles
        self.SLSignal = [0] * len(self.df)

    def calculate_SLSignal(self):
        for row in range(self.SLbackcandles, len(self.df)):
            mi = 1e10
            ma = -1e10
            if self.df.EMAsignal[row] == 1:
                for i in range(row - self.SLbackcandles, row + 1):
                    ma = max(ma, self.df.High[i])
                self.SLSignal[row] = ma
            if self.df.EMAsignal[row] == 2:
                for i in range(row - self.SLbackcandles, row + 1):
                    mi = min(mi, self.df.Low[i])
                self.SLSignal[row] = mi

        self.df['SLSignal'] = self.SLSignal

class OrderSignal:
    def __init__(self, df):
        self.df = df

    def totalSignal(self):
        ordersignal = [0] * len(self.df)
        for i in range(len(self.df)):
            if (self.df.EMA20[i] > self.df.EMA50[i] and
                self.df.Heiken_Open[i] < self.df.EMA20[i] and
                self.df.Heiken_Close[i] > self.df.EMA20[i]):
                ordersignal[i] = 2
            if (self.df.EMA20[i] < self.df.EMA50[i] and
                self.df.Heiken_Open[i] > self.df.EMA20[i] and
                self.df.Heiken_Close[i] < self.df.EMA20[i]):
                ordersignal[i] = 1
        self.df['ordersignal'] = ordersignal

class StopLossCalculator:
    def __init__(self, df, SLbackcandles=1):
        self.df = df
        self.SLSignal = [0] * len(self.df)
        self.SLbackcandles = SLbackcandles
    
    def calculate_stop_loss(self):
        for row in range(self.SLbackcandles, len(self.df)):
            mi = 1e10
            ma = -1e10
            if self.df.ordersignal[row] == 1:
                for i in range(row - self.SLbackcandles, row + 1):
                    ma = max(ma, self.df.High[i])
                self.SLSignal[row] = ma
            if self.df.ordersignal[row] == 2:
                for i in range(row - self.SLbackcandles, row + 1):
                    mi = min(mi, self.df.Low[i])
                self.SLSignal[row] = mi
        
        self.df['SLSignal_heiken'] = self.SLSignal
        return self.df

class SignalGenerator:
    def __init__(self, df):
        self.df = df
        self.length = len(df)
        self.High = list(df['High'])
        self.Low = list(df['Low'])
        self.Close = list(df['Close'])
        self.Open = list(df['Open'])

    def Revsignal1(self):
        length = self.length
        High = self.High
        Low = self.Low
        Close = self.Close
        Open = self.Open

        signal = [0] * length
        bodydiff = [0] * length
        bodydiffmin = 0.003

        for row in range(1, length):
            bodydiff[row] = abs(Open[row] - Close[row])
            if (bodydiff[row] > bodydiffmin and bodydiff[row - 1] > bodydiffmin and
                Open[row - 1] < Close[row - 1] and
                Open[row] > Close[row] and
                (Open[row] - Close[row - 1]) >= +0e-5 and Close[row] < Open[row - 1]):
                signal[row] = 1
            elif (bodydiff[row] > bodydiffmin and bodydiff[row - 1] > bodydiffmin and
                  Open[row - 1] > Close[row - 1] and
                  Open[row] < Close[row] and
                  (Open[row] - Close[row - 1]) <= -0e-5 and Close[row] > Open[row - 1]):
                signal[row] = 2
            else:
                signal[row] = 0

            # Random signal generation for testing purposes
            # signal[row] = random.choice([0, 1, 2])

        return signal

    def generate_signals(self):
        self.df['signal1'] = self.Revsignal1()
        return self.df

class TrendAnalyzer:
    def __init__(self, df):
        self.df = df
        self.length = len(df)
        self.High = list(df['High'])
        self.Low = list(df['Low'])
        self.Close = list(df['Close'])
        self.Open = list(df['Open'])

    def mytarget(self, barsfront):
        length = self.length
        High = self.High
        Low = self.Low
        Close = self.Close
        Open = self.Open

        trendcat = [None] * length
        piplim = 300e-5

        for line in range(0, length - 1 - barsfront):
            for i in range(1, barsfront + 1):
                if ((High[line + i] - max(Close[line], Open[line])) > piplim) and ((min(Close[line], Open[line]) - Low[line + i]) > piplim):
                    trendcat[line] = 3  # no trend
                elif (min(Close[line], Open[line]) - Low[line + i]) > piplim:
                    trendcat[line] = 1  # downtrend
                    break
                elif (High[line + i] - max(Close[line], Open[line])) > piplim:
                    trendcat[line] = 2  # uptrend
                    break
                else:
                    trendcat[line] = 0  # no clear trend

        return trendcat

    def evaluate_results(self, trendId, signal_column):
        conditions = [(self.df['Trend'] == 1) & (self.df[signal_column] == 1),
                      (self.df['Trend'] == 2) & (self.df[signal_column] == 2)]
        values = [1, 2]
        self.df['result'] = np.select(conditions, values)

        # Calculate accuracy
        trend_count = self.df[self.df['result'] == trendId].shape[0]
        signal_count = self.df[self.df[signal_column] == trendId].shape[0]
        accuracy = trend_count / signal_count if signal_count > 0 else 0

        return accuracy

    def get_false_positives(self, trendId, signal_column):
        false_positives = self.df[(self.df['Trend'] != trendId) & (self.df[signal_column] == trendId)]
        return false_positives

    def run_evaluation(self, barsfront, trendId, signal_column):
        self.df['Trend'] = self.mytarget(barsfront)
        accuracy = self.evaluate_results(trendId, signal_column)
        false_positives = self.get_false_positives(trendId, signal_column)
        return accuracy, false_positives

class SignalGenerator1:
    def __init__(self, df):
        self.df = df

    def generate_bollinger_bands(self, window=20, window_dev=2):
        # Calculate 20-period moving average
        self.df['ma20'] = self.df['Close'].rolling(window=window).mean()

        # Fill missing values in the 'ma20' column with the mean of the column
        self.df['ma20'].fillna(self.df['ma20'].bfill(), inplace=True)

        # Initialize Bollinger Bands Indicator
        indicator_bb = BollingerBands(close=self.df["Close"], window=window, window_dev=window_dev)

        # Add Bollinger Bands features
        self.df['middle_band'] = indicator_bb.bollinger_mavg()
        self.df['upper_band'] = indicator_bb.bollinger_hband()
        self.df['lower_band'] = indicator_bb.bollinger_lband()

        # Fill missing values in Bollinger Bands columns with the mean of the respective columns
        self.df['middle_band'].fillna(self.df['middle_band'].bfill(), inplace=True)
        self.df['upper_band'].fillna(self.df['upper_band'].bfill(), inplace=True)
        self.df['lower_band'].fillna(self.df['lower_band'].bfill(), inplace=True)

    def generate_candlestick_signals(self):
        # Buy when the price breaks above the upper Bollinger Band
        self.df['buy_signal'] = self.df['Close'] > self.df['upper_band']
        self.df.loc[self.df['buy_signal'] == True, 'Position'] = 1

        # Sell when the price drops below the 20-day moving average
        self.df['sell_signal'] = self.df['Close'] < self.df['ma20']
        self.df.loc[self.df['sell_signal'] == True, 'Position'] = -1

        # Forward fill the position column to ensure we hold onto our position until a sell signal is generated
        self.df['Position'].fillna(method='ffill', inplace=True)

        # Calculate the daily returns of the strategy
        self.df['Strategy Returns'] = self.df['Close'].pct_change() * self.df['Position'].shift(1)

    def generate_fractal_signals(self):
        i = Indicators(self.df)
        i.fractals(column_name_high='fractal_high', column_name_low='fractal_low')
        self.df = i.df

        # Buy when the price breaks above the fractal high
        self.df['buy_signal1'] = self.df['Close'] > self.df['fractal_high']
        self.df.loc[self.df['buy_signal1'] == True, 'Position'] = 1

        # Sell when the price drops below the fractal low
        self.df['sell_signal1'] = self.df['Close'] < self.df['fractal_low']
        self.df.loc[self.df['sell_signal1'] == True, 'Position'] = -1

        # Forward fill the position column to ensure we hold onto our position until a sell signal is generated
        self.df['Position'].fillna(method='ffill', inplace=True)

        # Calculate the daily returns of the strategy
        self.df['Strategy Returns'] = self.df['Close'].pct_change() * self.df['Position'].shift(1)

    def generate_combined_signals(self):
        # Volume Signal
        VSignal = [0] * len(self.df)
        vbackcandles = 1
        for row in range(vbackcandles + 1, len(self.df)):
            VSignal[row] = 1
            for i in range(row - vbackcandles, row):
                if self.df.Volume[row] < self.df.Volume[i] and self.df.Volume[row - 1] < self.df.Volume[row - 2]:
                    VSignal[row] = 0
        self.df['VSignal'] = VSignal

        # Price Signal
        PriceSignal = [0] * len(self.df)
        pbackcandles = 4
        for row in range(pbackcandles, len(self.df)):
            PriceSignal[row] = 1
            for i in range(row - pbackcandles, row):
                if self.df.EMASignal[row] == 1:  # downtrend
                    if self.df.Open[row] <= self.df.Close[row]:  # downcandle row
                        PriceSignal[row] = 0
                    elif self.df.Open[i] > self.df.Close[i]:  # downcandle i we are looking for 4 upcandles
                        PriceSignal[row] = 0
                if self.df.EMASignal[row] == 2:  # uptrend
                    if self.df.Open[row] >= self.df.Close[row]:  # upcandle row
                        PriceSignal[row] = 0
                    elif self.df.Open[i] < self.df.Close[i]:  # upcandle i we are looking for 4 dowcandles
                        PriceSignal[row] = 0
                else:
                    PriceSignal[row] = 0
        self.df['PriceSignal'] = PriceSignal

        # Total Signal
        TotSignal = [0] * len(self.df)
        for row in range(0, len(self.df)):
            if self.df.EMASignal[row] == 1 and self.df.VSignal[row] == 1 and self.df.PriceSignal[row] == 1:
                TotSignal[row] = 1
            if self.df.EMASignal[row] == 2 and self.df.VSignal[row] == 1 and self.df.PriceSignal[row] == 1:
                TotSignal[row] = 2
        self.df['TotSignal'] = TotSignal

        # Stop Loss Signal
        SLSignal = [0] * len(self.df)
        SLbackcandles = 4
        for row in range(SLbackcandles, len(self.df)):
            mi = 1e10
            ma = -1e10
            if self.df.EMASignal[row] == 1:
                for i in range(row - SLbackcandles, row + 1):
                    ma = max(ma, self.df.High[i])
                SLSignal[row] = ma
            if self.df.EMASignal[row] == 2:
                for i in range(row - SLbackcandles, row + 1):
                    mi = min(mi, self.df.Low[i])
                SLSignal[row] = mi

        self.df['SLSignal'] = SLSignal

    def generate_signals(self):
        self.generate_bollinger_bands()
        self.generate_candlestick_signals()
        self.generate_fractal_signals()
        self.generate_combined_signals()

        return self.df

import numpy as np

class GridSignalGenerator:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def generate_grid(midprice, grid_distance, grid_range):
        return np.arange(midprice - grid_range, midprice + grid_range, grid_distance)

    def generate_grid_signal(self, grid_distance=0.005, grid_range=0.1):
        midprice = self.df['High'].median()
        grid = self.generate_grid(midprice=midprice, grid_distance=grid_distance, grid_range=grid_range)
        signal = [0] * len(self.df)

        for index, row in self.df.iterrows():
            for p in grid:
                if min(row.Low, row.High) < p and max(row.Low, row.High) > p:
                    signal[index] = 1

        self.df["grid_signal"] = signal

    def generate_heiken_ashi(self):
        self.df['Heiken_Close'] = (self.df.Open + self.df.Close + self.df.High + self.df.Low) / 4
        self.df['Heiken_Open'] = self.df['Open']
        for i in range(1, len(self.df)):
            self.df['Heiken_Open'][i] = (self.df.Heiken_Open[i - 1] + self.df.Heiken_Close[i - 1]) / 2

        self.df['Heiken_High'] = self.df[['High', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
        self.df['Heiken_Low'] = self.df[['Low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)

    def generate_emas_and_rsi(self):
        import pandas_ta as ta

        # Calculate Exponential Moving Averages (EMAs) and RSI
        self.df["EMA20"] = ta.ema(self.df.Close, length=20)
        self.df["EMA50"] = ta.ema(self.df.Close, length=50)
        self.df['RSI'] = ta.rsi(self.df.Close, length=12)

        # Fill missing values for the specified columns using backfill
        columns_to_fill = ["EMA20", "EMA50", "RSI"]
        self.df[columns_to_fill] = self.df[columns_to_fill].fillna(method='bfill')

    def generate_signal(self):
        self.generate_grid_signal()
        self.generate_heiken_ashi()
        self.generate_emas_and_rsi()

        # Signal based on EMAs and Heiken Ashi
        ordersignal = [0] * len(self.df)
        for i in range(0, len(self.df)):
            if (self.df.EMA20[i] > self.df.EMA50[i] and self.df.Heiken_Open[i] < self.df.EMA20[i]
                    and self.df.Heiken_Close[i] > self.df.EMA20[i]):
                ordersignal[i] = 2
            if (self.df.EMA20[i] < self.df.EMA50[i] and self.df.Heiken_Open[i] > self.df.EMA20[i]
                    and self.df.Heiken_Close[i] < self.df.EMA20[i]):
                ordersignal[i] = 1
        self.df['ordersignal'] = ordersignal

        # StopLoss from signal
        SLSignal = [0] * len(self.df)
        SLbackcandles = 1
        for row in range(SLbackcandles, len(self.df)):
            mi = 1e10
            ma = -1e10
            if self.df.ordersignal[row] == 1:
                for i in range(row - SLbackcandles, row + 1):
                    ma = max(ma, self.df.High[i])
                SLSignal[row] = ma
            if self.df.ordersignal[row] == 2:
                for i in range(row - SLbackcandles, row + 1):
                    mi = min(mi, self.df.Low[i])
                SLSignal[row] = mi

        self.df['SLSignal_heiken'] = SLSignal

class MartiangleSignal:
    def __init__(self, df, backcandles=6):
        self.df = df
        self.backcandles = backcandles

    def add_emasignal(self):
        emasignal = [0] * len(self.df)
        for row in range(self.backcandles, len(self.df)):
            upt = 1
            dnt = 1
            for i in range(row - self.backcandles, row + 1):
                if self.df.High[i] >= self.df.EMA[i]:
                    dnt = 0
                if self.df.Low[i] <= self.df.EMA[i]:
                    upt = 0
            if upt == 1 and dnt == 1:
                emasignal[row] = 3
            elif upt == 1:
                emasignal[row] = 2
            elif dnt == 1:
                emasignal[row] = 1
        self.df['EMASignal1'] = emasignal


    def total_signal(self, adx_threshold=20, rsi_threshold=25):
        ordersignal = [0] * len(self.df)
        self.df['ADX_14'] = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)['ADX_14']
        for i in range(len(self.df)):
            if self.df.RSI[i] <= rsi_threshold and self.df.ADX_14[i] >= adx_threshold and self.df.EMASignal1[i] == 2:
                ordersignal[i] = 2
        self.df['long_signal'] = ordersignal

    def add_martiangle_signal(self):
        self.df['martiangle_signal'] = np.random.random(len(self.df))
        self.df['martiangle_signal'] = self.df['martiangle_signal'].apply(lambda x: 1 if x < 0.5 else 2)

    def calculate_signals(self, adx_threshold=20, rsi_threshold=25):
        self.add_emasignal()
        self.total_signal(adx_threshold, rsi_threshold)
        self.add_martiangle_signal()

class DataProcessor:
    def __init__(self, df):
        self.df = df


    def fill_missing_values(self):
        self.df['Trend'].fillna('4.0', inplace=True)
        self.df['Strategy Returns'].dropna(inplace=True)

    def process_candle_direction(self):
        # Calculate the difference between the current 'Close' and the previous 'Close'
        self.df['Close_diff'] = self.df['Close'].diff()

        # Create a new column indicating whether the candle is up or down from the previous one
        self.df['Candle_direction'] = 'Up'  # By default, set all candles to 'Up'

        # Set 'Candle_direction' to 'Down' for candles with a negative 'Close_diff'
        self.df.loc[self.df['Close_diff'] < 0, 'Candle_direction'] = 'Down'

        # Replace 'Up' by 1 and 'Down' by 2
        self.df['Candle_direction'].replace({'Up': 1, 'Down': 2}, inplace=True)

        # Drop the 'Close_diff' column if you don't need it anymore
        self.df.drop(columns=['Close_diff'], inplace=True)
    
    '''def process_candle_direction(self):
    # Calculate the difference between the next 'Close' and the current 'Close'
        self.df['Next_Close_diff'] = self.df['Close'].shift(-1) - self.df['Close']

    # Create a new column 'Candle_direction' that indicates if the next candle is higher
        self.df['Candle_direction'] = (self.df['Next_Close_diff'] < 0).astype(int)

    # Drop the 'Next_Close_diff' column if you don't need it anymore
        self.df.drop(columns=['Next_Close_diff'], inplace=True)'''


    def clean_dataframe(self):
        columns_to_drop = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMA',
            'ma20', 'middle_band', 'upper_band', 'lower_band', 'Strategy Returns',
            'Heiken_Close', 'Heiken_Open', 'Heiken_High', 'Heiken_Low', 'EMA20',
            'EMA50', 'ADX_14', 'pointpos', 'Channel', 'Trend'
        ]
        self.df_cleaned = self.df.drop(columns=columns_to_drop, axis='columns')

    def replace_candle_direction(self):
        self.df_cleaned['Candle_direction'] = self.df_cleaned['Candle_direction'].replace(2, 0)
        self.df_cleaned['Candle_direction'] = self.df_cleaned['Candle_direction'].replace(2, 1)

    def convert_boolean_to_int(self):
        boolean_columns = [
            'buy_signal', 'sell_signal', 'fractal_high', 'fractal_low',
            'buy_signal1', 'sell_signal1', 'fractals_high', 'fractals_low'
        ]
        for col in boolean_columns:
            # Fill missing values with False (0) before converting to integers
            self.df_cleaned[col].fillna(False, inplace=True)
            self.df_cleaned[col] = self.df_cleaned[col].astype(int)

    def replace_position_values(self):
        self.df_cleaned['Position'] = self.df_cleaned['Position'].replace({-1: 1, 1: 2})
        self.df_cleaned.rename(columns={'EMAsignal': 'EMASignal'}, inplace=True)


    def process_columns_based_on_mean(self):
        mean_column1 = self.df_cleaned['fractal_high'].mean()
        mean_column2 = self.df_cleaned['fractal_low'].mean()
        mean_column3 = self.df_cleaned['fractals_high'].mean()
        mean_column4 = self.df_cleaned['fractals_low'].mean()
        mean_column5 = self.df_cleaned['SLSignal_heiken'].mean()

        # Assign 1 to values in columns greater than their respective means, and 0 to the rest
        self.df_cleaned['fractal_high'] = self.df_cleaned['fractal_high'].apply(lambda x: 1 if x > mean_column1 else 0)
        self.df_cleaned['fractal_low'] = self.df_cleaned['fractal_low'].apply(lambda x: 1 if x > mean_column2 else 0)
        self.df_cleaned['fractals_high'] = self.df_cleaned['fractals_high'].apply(lambda x: 1 if x > mean_column3 else 0)
        self.df_cleaned['fractals_low'] = self.df_cleaned['fractals_low'].apply(lambda x: 1 if x > mean_column4 else 0)
        self.df_cleaned['SLSignal_heiken'] = self.df_cleaned['SLSignal_heiken'].apply(lambda x: 1 if x > mean_column5 else 0)

    def save_to_csv(self):
        self.df_cleaned.to_csv(os.path.join('artifacts', 'df_cleaned.csv'), index=False)

    def process_all(self):
        self.fill_missing_values()
        self.process_candle_direction()
        self.clean_dataframe()
        self.replace_candle_direction()
        self.convert_boolean_to_int()
        self.replace_position_values()
        self.process_columns_based_on_mean()
        self.save_to_csv()


class StockDataPipeline:
    def __init__(self, ticker, period, interval):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df_new = None

    def run_pipeline(self):
        # Download data
        downloader = StockDataDownloader(self.ticker, self.period, self.interval)
        df = downloader.download_data()

        self.df_new = df.copy()

        # Preprocess data
        preprocessor = DataPreprocessor(df)
        df = preprocessor.remove_zero_volume()
        df = preprocessor.add_time_columns()

        # Calculate technical indicators
        indicators = TechnicalIndicators(df)
        df = indicators.calculate_rsi()
        df = indicators.calculate_ema()

        # Generate EMA signals
        signal_generator2 = SignalGenerator2(df)
        df = signal_generator2.generate_ema_signal()

        # Generate pivot signals
        pivot_generator = PivotSignalGenerator(df)
        df = pivot_generator.generate_pivot_signals()

        # Generate CHOCH pattern
        choch_detector = CHOCHPatternDetector(df)
        df = choch_detector.generate_choch_pattern()

        # Generate Fibonacci signals
        fibonacci_generator = FibonacciSignalGenerator(df)
        df = fibonacci_generator.generate_fibonacci_signals()

        # Generate Level Break signals
        level_break_detector = LevelBreakSignalDetector(df)
        df = level_break_detector.generate_level_break_signals()

        # Generate Support and Resistance signals
        support_resistance_detector = SupportResistanceSignalDetector(df)
        df = support_resistance_detector.generate_support_resistance_signals()

        # Generate Channel signals
        channel_detector = ChannelDetector(df, backcandles=40, window=15)
        df = channel_detector.generate_channel_signals()

        # Generate Breakout signals
        breakout_detector = BreakoutDetector(df, backcandles=40, window=15)
        df = breakout_detector.detect_breakouts()

        # Generate Candlestick patterns
        candlestick_detector = CandlestickPatternDetector(df)
        df = candlestick_detector.detect_candlestick_patterns()

        trend_targeting = TrendTargeting(df, barsfront=3)
        trend_targeting.get_trend_categories()

        trend_resultant = TrendResultant(df)
        trend_resultant.apply_conditions()

        engulfing_pattern_detector = EngulfingPatternDetector(df)
        engulfing_pattern_detector.get_signal_counts()

        tech_ind = BollingerIndicators(df)
        tech_ind.calculate_moving_average(window=20)
        tech_ind.calculate_bollinger_bands(window=20, window_dev=2)

        bollinger_band_strategy = BollingerBandStrategy(df)
        bollinger_band_strategy.calculate_signals()

        fractal_indicator = FractalStrategy(df)
        fractal_indicator.calculate_fractal()
        fractal_indicator.calculate_fractals()
        fractal_indicator.generate_signals()

        v_signal = VSignalGenerator(df)
        v_signal.add_vsignal_column()

        price_signal = PriceSignalGenerator(df, pbackcandles=4)
        price_signal.generate_price_signal()

        tot_signal = SignalProcessor(df)
        tot_signal.get_tot_signal()

        sl_signal = SLSignalCalculator(df)
        sl_signal.calculate_SLSignal()

        # Generate Grid signals
        grid_signal_generator = GridSignalGenerator(df)
        grid_signal_generator.generate_grid_signal()

        # Generate Heiken Ashi and EMAs
        heiken_ashi_generator = GridSignalGenerator(df)
        heiken_ashi_generator.generate_heiken_ashi()
        heiken_ashi_generator.generate_emas_and_rsi()

        order_signal = OrderSignal(df)
        order_signal.totalSignal()

        sl_signal_heiken = StopLossCalculator(df, SLbackcandles=1)
        sl_signal_heiken.calculate_stop_loss()

        # Generate Martiangle signals
        martiangle_signal_generator = MartiangleSignal(df)
        martiangle_signal_generator.calculate_signals()

        data_processor = DataProcessor(df)
        data_processor.process_all()

        self.df_new.to_csv(os.path.join('artifacts', 'df_new.csv'), index=True)

