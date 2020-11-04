# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 08:53:30 2020

@author: LStue
"""

""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('Durchflussmessung')['QREF']
X = np.array(data).reshape(-1)

#a) Absolute Häufigkeit
fig1, ax1 = plt.subplots()
ax1.hist(X)
ax1.set_xlabel('Durchfluss $Q/\mathrm{m}^3/\mathrm{h}$')
ax1.set_ylabel('Absolute Häufigkeit')
ax1.set_title('Histogramm zur Durchflussmessung')
ax1.grid(True)

#b) CI
X_quer = np.mean(X)
s = np.std(X,ddof=1)
N = np.size(X)

ci_mu = stats.t.interval(0.95,N-1,loc=X_quer,scale=s/np.sqrt(N))
ci_sigma = stats.chi2.interval(0.95,N-1,loc=0,scale=s/(N))

#c)Histogramm2
fig2, ax2 = plt.subplots()
ax2.hist(X, density='true')
x_axis = np.linspace(0.45,0.55,1000)
pdf = stats.norm.pdf(x_axis,loc=X_quer,scale=s)
ax2.plot(x_axis,pdf)

#d)Hypothesentest: Grundgesamtheit der Stichprobe hat Mittelwert 0.5
test1 = stats.ttest_1samp(X,0.5)
