# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:13:09 2020

@author: LStue
"""

""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('<Dateiname>')['data']
X = np.array(data).reshape(-1)


# Histogramme
test=np.array([1,2,2,3,3,3,4,4,4,4])
#absolute Häufigkeit
fig1, ax1 = plt.subplots()
ax1.hist(test)
# relative Häufigkeit
fig2, ax2 = plt.subplots()
ax2.hist(test, weights=np.zeros_like(test) + 1. / test.size)
# Wahrscheinlichkeitsverteilung
fig3, ax3 = plt.subplots()
ax3.hist(test, density='true')