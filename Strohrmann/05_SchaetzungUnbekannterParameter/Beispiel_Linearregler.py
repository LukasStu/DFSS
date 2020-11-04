# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""

""" Bibliotheken importieren"""
from scipy.stats import norm     # Normalverteitung
import numpy as np

"""Werte aus Aufgabe übernehmen"""
gamma = 0.95;
N = 100
dataquer = 5
s = np.sqrt(0.09)

print(' ')
print('Mittelwert x: ', dataquer)
print('Standardabweichung s: ', s)

c1 = norm.ppf((1-gamma)/2)
c2 = norm.ppf((1+gamma)/2)
mu_min = dataquer - ((c2*s)/np.sqrt(N))
mu_max = dataquer - ((c1*s)/np.sqrt(N))

print(' ')
print('Untere Grenze für Mittelwert \u03bc: ', mu_min)
print('Obere Grenze für Mittelwert \u03bc:', mu_max)
