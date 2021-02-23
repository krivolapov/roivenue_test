# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:38:43 2021

@author: Maksim Krivolapov
"""

from BudgetOptimizer import EstimatorClass, settings_estimator


import pandas as pd
#pd.set_option()
import sys
import json

encoding = sys.getdefaultencoding()
import os
import numpy as np
import xlsxwriter
from pathlib import Path


pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.3f}'.format

from BudgetOptimizer import EstimatorClass, settings_estimator
from BudgetOptimizer import DateTimeID, dtype_, stat_name, regr_name, r2_stat_name, usecols_

list_files = os.listdir("./input")
mypath = Path().absolute()
folder = 'budget_opt\\'
img_folder = folder + 'img\\'

#################################################################################

df = pd.read_csv('input\\'+ list_files[0], sep="\t", infer_datetime_format=True,
                 usecols=usecols_, dtype=dtype_, parse_dates=DateTimeID)

df['periodStartDate'] = pd.to_datetime(df['periodStartDate'])

estimator = EstimatorClass()

estimator.load_settings(settings_estimator)

estimator.load_data(df)

result = estimator.corr_data()