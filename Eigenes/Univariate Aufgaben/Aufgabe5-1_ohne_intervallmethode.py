# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:41:14 2020

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
from scipy.io import loadmat # Für mat-Dateien
from scipy import stats


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('AusdehnungKunststoff')['AusdehnungKunststoff']
x = np.array(data).reshape(-1)

"a) 95% und 99%-Konfidenzintervall für mu"
x_quer = np.mean(x)
s = np.std(x,ddof=1)
N = x.size

print(' ')
print('Mittelwert x: ', x_quer)
print('Standardabweichung s: ', s)

# Konfidenzzahlen
gamma95 = 0.95
gamma99 = 0.99 
# Grenzen für 95%
c1_95 = stats.t.ppf((1-gamma95)/2,N-1)
c2_95 = stats.t.ppf((1+gamma95)/2,N-1)
# Grenzen für 99%
c1_99 = stats.t.ppf((1-gamma99)/2,N-1)
c2_99 = stats.t.ppf((1+gamma99)/2,N-1)
# 95% Konfidenzintervall
mu_min_95 = x_quer-c2_95*s/np.sqrt(N)
mu_max_95 = x_quer+c2_95*s/np.sqrt(N)
# 99% Konfidenzintervall
mu_min_99 = x_quer-c2_99*s/np.sqrt(N)
mu_max_99 = x_quer+c2_99*s/np.sqrt(N)

print(' ')
print('Untere Grenze für Mittelwert \u03bc, 95%:', mu_min_95)
print('Obere Grenze für Mittelwert \u03bc, 95%:', mu_max_95)

print(' ')
print('Untere Grenze für Mittelwert \u03bc, 99%: ', mu_min_99)
print('Obere Grenze für Mittelwert \u03bc, 99%:', mu_max_99)

"b) Zweiseitges Konfidenzintervall für sigma"
# Grenzen für 95%
c1_95_sig = stats.chi2.ppf((1-gamma95)/2,N-1)
c2_95_sig = stats.chi2.ppf((1+gamma95)/2,N-1)
# 95% Konfidenzintervall
sigma_min_95 = np.sqrt((N-1)/c2_95_sig)*s
sigma_max_95 = np.sqrt((N-1)/c1_95_sig)*s
print(' ')
print('Untere Grenze für Standardabweichung: ', sigma_min_95)
print('Obere Grenze für Standardabweichung: ', sigma_max_95)

"Prognosebereich delta x mit 99,73%"
c1_9973 = stats.t.ppf((1-0.9973)/2,N-1)
c2_9973 = stats.t.ppf((1+0.9973)/2,N-1)

deltax_min = x_quer+c1_9973*s*np.sqrt(1+1/N)
deltax_max = x_quer+c2_9973*s*np.sqrt(1+1/N)
print(' ')
print('Untere Progrnosegrenze für delta x: ', deltax_min)
print('Obere Progrnosegrenze für delta x: ', deltax_max)

