# -*- coding: utf-8 -*-
"""
Created on 18.01.2012

@author: Maksim Krivolapov
"""

import numpy as np
from scipy.optimize import differential_evolution
import warnings

def main_statistics(df, precision = 3):
    """ Function input pandas series or numpy array
        input: dataframe, precision -> digits after point
        output: mean, std, median, skewness, kurtosis, variance, interquartile range, Shapiro-Wilk Test [Stat, Pr] """
    from scipy.stats import kurtosis, iqr, shapiro
    result = []
    count,mean, std, *all = df.describe()
    result.append(np.round(mean, precision))
    result.append(np.round(std, precision))
    result.append(np.round(df.median(), precision))
    result.append(np.round(df.skew(axis = 0, skipna = True), precision ))
    result.append(np.round(kurtosis(df, fisher=True), precision ))
    result.append(np.round(df.var(), precision))
    result.append((np.round(iqr(df, axis=0, keepdims=True), precision)).item())
    if len(df) > 2:
        stat, p = shapiro(df)
    else:
        stat = np.nan
        p = np.nan
    result.append(np.round(stat,precision))
    result.append(np.round(p,precision))
    return result

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    function = regr_func # it's bad staff #TODO
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = function(t_train, *parameterTuple)
    return np.sum((y_train - val) ** 2.0)

def generate_Initial_Parameters(t_train, y_train,function):
    # min and max used for bounds
    maxX = max(t_train)
    minX = min(t_train)
    maxY = max(y_train)
    minY = min(y_train)
    maxXY = max(maxX, maxY)

    parameterBounds = []
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for a
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for b
    parameterBounds.append([-maxXY, maxXY]) # seach bounds for c

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# regression functions definition
def log_f(x, a, b, c):
    return a * (1-np.exp(-x/b)) + c #a * (1 - np.exp((x/b)))

def line_f(x, a, b):
    return a * x + b

def sine_f(x, a, b):
    return a * np.sin(b * x)

