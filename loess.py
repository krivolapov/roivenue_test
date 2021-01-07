# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:43:46 2021

@author: Max
"""

import numpy as np
import pylab as plt
from skmisc.loess import loess

x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.4

l = loess(x,y)
l.fit()
pred = l.predict(x, stderror=True)
conf = pred.confidence()

lowess = pred.values
ll = conf.lower
ul = conf.upper

plt.plot(x, y, '+')
plt.plot(x, lowess)
plt.fill_between(x,ll,ul,alpha=.33)
plt.show()