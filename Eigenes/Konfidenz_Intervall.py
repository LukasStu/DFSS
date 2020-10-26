# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:17:08 2020

@author: lukas
"""

from scipy import stats
import numpy as np

# Hier t-Verteilung implizit
a = np.array([1,2,3,4,4,4,5,5,5,5,4,4,4,6,7,8])
x_quer, s = np.mean(a), np.std(a)
conf_int1 = stats.norm.interval(0.68, loc=x_quer, scale=s/np.sqrt(len(a)))

#print('{:0.2%} of the single draws are in conf_int_a'
#      .format(((a >= conf_int_a[0]) & (a < conf_int_a[1])).sum() / float(N)))

# Hier Standartnormalverteilung implizit
u_quer = 5.0
sigma = np.sqrt(0.09)
conf_int2 = stats.norm.interval(0.95, loc=u_quer, scale=sigma/np.sqrt(100))