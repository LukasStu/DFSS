# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:17:43 2020

@author: LStue
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
data = loadmat('Selbstentzuendung')['Temp']
x = np.array(data).reshape(-1)

N = x.size
x_quer = np.mean(x)
s = np.std(x,ddof=1)
print(' ')
print('Mittelwert x: ', x_quer)
print('Standardabweichung s: ', s)

"a) 95% Konfidenzintervall von mu und sigma"
gamma = 0.95
    # Grenzen c1 und c2 für mu, t-verteilt
c_mu = stats.t.interval(gamma,N-1)
    # 95% Konfidenzintervall
mu_min_95 = x_quer-c_mu[1]*s/np.sqrt(N)
mu_max_95 = x_quer-c_mu[0]*s/np.sqrt(N)
print(' ')
print('Untere Grenze für Mittelwert \u03bc, 95%:', mu_min_95)
print('Obere Grenze für Mittelwert \u03bc, 95%:', mu_max_95)
    # Grenzen c1 und c2 für sigma, chi²-verteilt
c_sigma = stats.chi2.interval(gamma,N-1)
    # 95% Konfidenzintervall
sigma_min_95 = np.sqrt((N-1)/c_sigma[1])*s
sigma_max_95 = np.sqrt((N-1)/c_sigma[0])*s
print(' ')
print('Untere Grenze für Standardabweichung: ', sigma_min_95)
print('Obere Grenze für Standardabweichung: ', sigma_max_95)

"b) Histogramm 1"
#absolute Häufigkeit
fig1, ax1 = plt.subplots()
ax1.hist(x,bins=int(np.sqrt(N)))
ax1.set_xlabel(r'Selbstentzündungstemperatur $\dfrac{\mathrm{T}}{C^{\circ}}$')
ax1.set_ylabel(r'Anzahl der Proben')
ax1.axvline(x=mu_min_95,color='r')
ax1.axvline(x=mu_max_95,color='r')

"b) Histogramm 1"
#absolute Häufigkeit
# Wahrscheinlichkeitsverteilung
fig2, ax2 = plt.subplots()
ax2.hist(x, density='true',label='Wahrscheinlichkeitsverteilung')
# Grundgesamtheit
xaxes = np.linspace(312,347,1000)
pdf = stats.norm.pdf(xaxes,loc=x_quer,scale=s)
ax2.plot(xaxes,pdf,'r',label='Wahrscheinlichkeitsdichte')

ax2.set_xlabel(r'Selbstentzündungstemperatur $\dfrac{\mathrm{T}}{C^{\circ}}$')
ax2.set_ylabel(r'Wahrscheinlichkeit')
ax2.legend()
