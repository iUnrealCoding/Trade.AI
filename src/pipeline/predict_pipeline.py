import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import tensorflow as tf

class PredictPipeline:
    def __init__(self, model_path=os.path.join('artifacts','model.h5')):
        self.model_path = model_path

    def predict(self, input_data):
        try:
            # Load model
            model = tf.keras.models.load_model(self.model_path)
            
            # Make predictions directly on input data
            EX = pd.read_csv(os.path.join('artifacts','test.csv'))
            X_test = EX.drop(['Candle_direction'],axis='columns')
            y_pred = model.predict(X_test)
            y_pred_series = pd.Series(y_pred.flatten())
            y_pred_mean=y_pred_series.mean()*0.1
            preds = model.predict(input_data)
            
            # Generate binary predictions based on mean threshold
            binary_preds = (preds > y_pred_mean).astype(int)

            return binary_preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Year, Month, Day, Hour, Minute, EMASignal, isPivot, CHOCH_pattern_detected, fibonacci_signal,
                 SL, TP, MinSwing, MaxSwing, LBD_detected, LBH_detected, SR_signal, isBreakOut, candlestick_signal,
                 result, signal1, buy_signal, Position, sell_signal, fractal_high, fractal_low, buy_signal1, 
                 sell_signal1, fractals_high, fractals_low, VSignal, PriceSignal, TotSignal, SLSignal, grid_signal,
                 ordersignal, SLSignal_heiken, EMASignal1, long_signal, martiangle_signal):
        
        self.Year = Year
        self.Month = Month
        self.Day = Day
        self.Hour = Hour
        self.Minute = Minute
        self.EMASignal = EMASignal
        self.isPivot = isPivot
        self.CHOCH_pattern_detected = CHOCH_pattern_detected
        self.fibonacci_signal = fibonacci_signal
        self.SL = SL
        self.TP = TP
        self.MinSwing = MinSwing
        self.MaxSwing = MaxSwing
        self.LBD_detected = LBD_detected
        self.LBH_detected = LBH_detected
        self.SR_signal = SR_signal
        self.isBreakOut = isBreakOut
        self.candlestick_signal = candlestick_signal
        self.result = result
        self.signal1 = signal1
        self.buy_signal = buy_signal
        self.Position = Position
        self.sell_signal = sell_signal
        self.fractal_high = fractal_high
        self.fractal_low = fractal_low
        self.buy_signal1 = buy_signal1
        self.sell_signal1 = sell_signal1
        self.fractals_high = fractals_high
        self.fractals_low = fractals_low
        self.VSignal = VSignal
        self.PriceSignal = PriceSignal
        self.TotSignal = TotSignal
        self.SLSignal = SLSignal
        self.grid_signal = grid_signal
        self.ordersignal = ordersignal
        self.SLSignal_heiken = SLSignal_heiken
        self.EMASignal1 = EMASignal1
        self.long_signal = long_signal
        self.martiangle_signal = martiangle_signal

    def get_data_as_array(self):
        try:
            return np.array([[self.Year, self.Month, self.Day, self.Hour, self.Minute, self.EMASignal, 
                              self.isPivot, self.CHOCH_pattern_detected, self.fibonacci_signal, self.SL, 
                              self.TP, self.MinSwing, self.MaxSwing, self.LBD_detected, self.LBH_detected, 
                              self.SR_signal, self.isBreakOut, self.candlestick_signal, self.result, 
                              self.signal1, self.buy_signal, self.Position, self.sell_signal, self.fractal_high, 
                              self.fractal_low, self.buy_signal1, self.sell_signal1, self.fractals_high, 
                              self.fractals_low, self.VSignal, self.PriceSignal, self.TotSignal, self.SLSignal, 
                              self.grid_signal, self.ordersignal, self.SLSignal_heiken, self.EMASignal1, 
                              self.long_signal, self.martiangle_signal]])
        except Exception as e:
            raise CustomException(e, sys)

