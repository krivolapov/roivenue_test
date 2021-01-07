# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:52:40 2021

@author: Max
"""

import numpy as np
import matplotlib.pyplot as plt
from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk

# generate 10 randomwalks of length 200
np.random.seed(33)
data = sim_randomwalk(n_series=10, timesteps=200, 
                      process_noise=10, measure_noise=30)

# operate smoothing
smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
smoother.smooth(data)

# generate intervals
low, up = smoother.get_intervals('prediction_interval', confidence=0.05)

# plot the first smoothed timeseries with intervals
plt.figure(figsize=(11,6))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)