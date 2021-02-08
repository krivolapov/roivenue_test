# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:38:43 2021

@author: Maksim Krivolapov
"""

from BudgetOptimizer import EstimatorClass, settings_estimator
import pandas as pd
from random import random


data = pd.DataFrame([random() for i in range(20)])

estimator = EstimatorClass()

estimator.load_settings(settings_estimator)

estimator.load_data(data)