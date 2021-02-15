#import pandas as pd
#import math
import matplotlib.pyplot as plt
#import numpy as np
#from scipy.stats import kurtosis, iqr, shapiro
#from scipy.optimize import differential_evolution
#import warnings
#import matplotlib.pyplot as plt

from Budget_optimiser_function import filter_df

settings_estimator = {'alpha'   :   0.1,
                      'corr_thd':   0.8,
                      'time_window_regr':12,
                      'period'  :   'week',
                      'margin'  :   1.0,
                      'level'   :   'platform',
                      'dependant_var':'grossProfit',
                      'independatnt_var': 'marketingInvestment',
                      'client': ''
                      }

class EstimatorClass:
    def __init__(self):
        self.dataset_size = 0
        pass

    def load_settings(self, settings):
        self.settings = settings
        pass

    def load_data(self, dataframe):
        self.data = dataframe
        self.dataset_size = len(self.data)
        print(type(self.data))

    def filter_data(self):
        df = self.data
        self.filtered_data = filter_df(df, self.settings['alpha'])
        pass

    def corr_data(self, data):
        pass


class OptimizerClass:
    def __init__(self):
        pass

    def load_settings(self, settings):
        self.settings = settings


class TrainerClass:
    def __init__(self):
        print("constructor")

    def load_settings(self, settings):
        self.settings = settings
        print("load settings")

    def get_settings(self):
        print(self.settings)

    def __start(self):
        print("private method")

    def call_private(self):
        self.__start()