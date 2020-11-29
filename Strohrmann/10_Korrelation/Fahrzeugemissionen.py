# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""


""" Bibliotheken importieren"""
import numpy as np
import pandas as pd
import scipy.stats  as stats
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import t, norm, chi2, f

""" Berechnung der Korrelation und es Hypothesentest für roh = 0 """
""" Stichprobenwerte aus Aufgabe übernehmen und als Dataframe speichern """

data = loadmat('Fahrzeugemissionen')
df = pd.DataFrame({'H': data['values'][:,0],
                   'P': data['values'][:,1],
                   'M': data['values'][:,2],
                   'E': data['values'][:,3]})

print(' ')
print(df)

""" Grafische Darstellung als Streudiagramm-Matrix"""

fig = plt.figure(1, figsize=(12,8))
fig.suptitle('')
ax1 = fig.subplots(1,1)
ax1= pd.plotting.scatter_matrix(df, alpha=1,Color='b', hist_kwds=dict(Color='b'))


""" Berechnung Korrelation und des Hypothesentests roh = 0 """

Corr = round(df.corr(method='pearson'),2)
print(' ')
print(Corr)

# data = df.to_numpy()
# Corr2, p = stats.pearsonr(data,axis=1)
# print(' ')
# print('Korrelation zwischen den Größen: ', Corr2)
# print('p-value zur Korrelation : ', round(p,5))

# """ Berechnung des Konfidenzbereichs """
# N = np.size(df['w'])
# gamma = 0.95
# c1 = norm.ppf((1-gamma)/2)
# c2 = norm.ppf((1+gamma)/2)
# roh1 = np.tanh(np.arctanh(Corr)-c2/np.sqrt(N-3))
# roh2 = np.tanh(np.arctanh(Corr)-c1/np.sqrt(N-3))

# print(' ')
# print('Untere Grenze des Korrelationskoeffizienten : ', round(roh1,3))
# print('Obere Grenze des Korrelationskoeffizienten : ', round(roh2,3))
