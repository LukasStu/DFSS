# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:12:05 2020

@author: LStue
"""

""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # Für mat-Dateien
from scipy import stats


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('AusdehnungKunststoff')['AusdehnungKunststoff']
X = np.array(data).reshape(-1)

# a) 95% und 99%-Konfidenzintervall für mu
x_quer = np.mean(X)
s = np.std(X)
N = X.size

ci95 = stats.t.interval(0.95,N-1,loc=x_quer,scale=s/np.sqrt(N))
print(ci95)
ci99 = stats.t.interval(0.99,N-1,loc=x_quer,scale=s/np.sqrt(N))
print(ci99)

# b) 95%-Konfidenzintervall der Standardabweichung


c1 = stats.chi2.ppf((1-0.95)/2,N-1)
c2 = stats.chi2.ppf((1+0.95)/2,N-1)

ci = np.array([((N-1)*np.square(s))/c2,((N-1)*np.square(s))/c1])
ci = np.sqrt(ci)
print(ci)
# b) 99.73%-Prognoseintervall des Mittelwerts

c11 = stats.t.ppf((1-0.9973)/2,N-1)
c22 = stats.t.ppf((1+0.9973)/2,N-1)

ci2 = np.array([x_quer+c11*s*np.sqrt(1+1/N),x_quer+c22*s*np.sqrt(1+1/N)])
print(ci2)