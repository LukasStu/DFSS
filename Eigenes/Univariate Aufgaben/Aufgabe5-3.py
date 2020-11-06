# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:16:37 2020

@author: LStue
"""

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data1 = loadmat('AbweichungenSoll')['Messung1']
data2 = loadmat('AbweichungenSoll')['Messung2']
x1 = np.array(data1).reshape(-1)
x2 = np.array(data2).reshape(-1)

N = x1.size
M = x2.size
x1_quer = np.mean(x1)
x2_quer = np.mean(x2)
s1 = np.std(x1,ddof=1)
s2 = np.std(x2,ddof=1)
s_gesamt = np.sqrt((s1**2*(N-1)+s2**2*(M-1))/(N+M-2))

print(' ')
print('Mittelwert x1: ', x1_quer)
print('Standardabweichung s1: ', s1)
print(' ')
print('Mittelwert x2: ', x2_quer)
print('Standardabweichung s2: ', s2)
print('Standardabweichung s_gesamt: ', s_gesamt)

"Konfidenzintervall"
gamma = 0.9973
c = stats.t.interval(gamma,N+M-2)
delta_mu_min = (x1_quer-x2_quer)-c[1]*np.sqrt(1/N+1/M)*s_gesamt
delta_mu_max = (x1_quer-x2_quer)-c[0]*np.sqrt(1/N+1/M)*s_gesamt
print(' ')
print('Untere Grenze für delta mu: ', delta_mu_min)
print('Obere Grenze für delta mu: ', delta_mu_max)