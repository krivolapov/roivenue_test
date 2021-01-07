# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:03:18 2020

@author: Max
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def f(x, start, end):
    res = np.empty_like(x)
    res[x < start] =-1
    res[x > end] = 1
    linear = np.all([[start <= x], [x <= end]], axis=0)[0]
    res[linear] = np.linspace(-1., 1., num=np.sum(linear))
    return res

if __name__ == '__main__':

    xdata = np.linspace(0., 1000., 1000)
    ydata = -np.ones(1000)
    ydata[500:1000] = 1.
    ydata = ydata + np.random.normal(0., 0.25, len(ydata))

    popt, pcov = curve_fit(f, xdata, ydata, p0=[495., 505.])
    print(popt, pcov)
    plt.figure()
    plt.plot(xdata, f(xdata, *popt), 'r-', label='fit')
    plt.plot(xdata, ydata, 'b-', label='data', alpha=0.5)
    plt.show()