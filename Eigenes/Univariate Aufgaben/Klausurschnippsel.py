# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:13:09 2020

@author: Lukas Stürmlinger Matrikel-Nummer:
"""

"""  Initialisierung: Variablen löschen, KOnsole leeren """    
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
data = loadmat('<Dateiname>')['data']
X = np.array(data).reshape(-1)

x_quer = np.mean(x)
s = np.std(x,ddof=1)
N = x.size

print(' ')
print('Mittelwert x: ', x_quer)
print('Standardabweichung s: ', s)


" Histogramme"
test=np.array([1,2,2,3,3,3,4,4,4,4])
#absolute Häufigkeit
fig1, ax1 = plt.subplots()
ax1.hist(test)
ax.set_xlabel(r'')
ax.set_ylabel(r'Absolute Häufigkeit')
# relative Häufigkeit
fig2, ax2 = plt.subplots()
ax2.hist(test, weights=np.zeros_like(test) + 1. / test.size)
ax.set_xlabel(r'')
ax.set_ylabel(r'Relative Häufigkeit')
# Wahrscheinlichkeitsverteilung
fig3, ax3 = plt.subplots()
ax3.hist(test, density='true')
ax.set_xlabel(r'')
ax.set_ylabel(r'Wahrscheinlichkeit')