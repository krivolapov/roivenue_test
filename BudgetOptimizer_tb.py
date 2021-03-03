# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:38:43 2021

@author: Maksim Krivolapov
"""


import pandas as pd
import sys
import os
from pathlib import Path
import pprint

encoding = sys.getdefaultencoding()
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.3f}'.format

from BudgetOptimizer import Optimizer
from BudgetOptimizer import DateTimeID, dtype_, usecols_

#####   Settings

list_files = os.listdir("./input")
mypath = Path().absolute()
folder = 'budget_opt\\'
img_folder = folder + 'img\\'

########################################################################

df = pd.read_csv('input\\'+ list_files[1], sep="\t", 
                 infer_datetime_format=True,
                 usecols=usecols_, 
                 dtype=dtype_, 
                 parse_dates=DateTimeID)

df['periodStartDate'] = pd.to_datetime(df['periodStartDate'])

result, stats = Optimizer(df, 
                          confidence_interval=3.0, 
                          change_investment = -200000, 
                          maximum_investment_change = 0.2)

print("\n Statistics of optimization \n")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(stats)


