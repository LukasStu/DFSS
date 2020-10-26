# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:45:40 2020

@author: lukas
Konfidenzintervall für die Varianz der Klebemenge
"""
from scipy import stats
import numpy as np

# Stichprobe
x = np.array([4.3, 4.5, 4.2, 4.3, 4.3, 4.7, 4.4, 4.2, 4.3, 4.5])
N = x.size

# Std.-Abw. der Stichprobe
s = np.std(x)

ci = stats.chi2.interval(0.95,N, scale=s)

c1 = stats.chi2.ppf((1-0.95)/2,N)
c2 = stats.chi2.ppf((1+0.95)/2,N)

ci2 = np.array([((N-1)*s**2)/c2,((N-1)*s**2)/c1])
ci2 = np.sqrt(ci2)