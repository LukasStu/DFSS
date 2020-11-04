# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""

""" Bibliotheken importieren"""
from scipy.stats import t     # Normalverteitung
from scipy.stats import chi2     # Normalverteitung
import numpy as np

"""Werte aus Aufgabe übernehmen"""
gamma = 0.95;
N = 10
m = [4.3, 4.5, 4.2, 4.3, 4.3, 4.7, 4.4, 4.2, 4.3, 4.5]
mquer = np.mean(m)
s = np.std(m,ddof=1)

"""Bstimmung der Kontanten C1 und c2 für den Mittelwert"""
c1 = t.ppf((1-gamma)/2,N-1)
c2 = t.ppf((1+gamma)/2,N-1)
mu_min = mquer - ((c2*s)/np.sqrt(N))
mu_max = mquer - ((c1*s)/np.sqrt(N))

print(' ')
print('Untere Grenze für Mittelwert \u03bc: ', mu_min)
print('Obere Grenze für Mittelwert \u03bc:', mu_max)

""" Bestimung der Konstanten für die Varianz """
c1 = chi2.ppf((1-gamma)/2,N-1)
c2 = chi2.ppf((1+gamma)/2,N-1)
sig_min = np.sqrt((N-1)/c2)*s
sig_max = np.sqrt((N-1)/c1)*s

print(' ')
print('Untere Grenze für Stabdardabweichung: ', sig_min)
print('Obere Grenze für Standardabweichung: ', sig_max)

ci = chi2.interval(gamma, N-1, loc=0, scale=1)