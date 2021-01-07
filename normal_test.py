# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:44:28 2021

@author: Max
"""

from scipy.stats import shapiro
import numpy as np
import matplotlib as plt

data = np.random.normal(loc = 20, scale = 5, size = 150)
stat, p = shapiro(data)
print(f'stat = {stat}, Pr = {p} \n')