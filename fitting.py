# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:37:42 2020

@author: Max
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
a = 2.0
b = 5.0
x = np.arange(0, 10, 0.1)

def function(x):
    return a*(1 - np.e**(-(x/b)))
    
y = function(x)

plt.plot(x,y)