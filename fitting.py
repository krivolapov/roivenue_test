# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:37:42 2020

@author: Max
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
a = 2.29
b = 1.0
x = np.arange(0, 10, 0.1)

def function(x):
    return a*(1 - np.exp(-(x/b)))
    
y = function(x)

plt.plot(x,y)
plt.savefig("test_fig.png",dpi=(250), bbox_inches='tight')
plt.close()

aaa = np.linspace(start = -10,stop =  10, num = 25)
yyy= np.arange(start = -1, stop =  1, step = 0.1)