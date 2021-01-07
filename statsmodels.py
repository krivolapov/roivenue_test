# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:02:27 2021

@author: Max
"""

import numpy as np
import pylab as plt
import statsmodels.api as sm

x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.2
lowess = sm.nonparametric.lowess(y, x, frac=0.1)

plt.plot(x, y, '+')
plt.plot(lowess[:, 0], lowess[:, 1])
plt.show()
