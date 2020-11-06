# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:34:14 2020

@author: LStue
"""


""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('Selbstentzuendung')['Temp']
X = np.array(data).reshape(-1)

N = X.size
X_quer = np.mean(X)
s_X_square = np.var(X,ddof=1)

gamma = 0.95

c1 = stats.t.ppf((1-gamma)/2,N-1)
c2 = stats.t.ppf((1+gamma)/2,N-1)

ci_mu_95 = (X_quer-c2*s_X_square/np.sqrt(N),X_quer-c1*s_X_square/np.sqrt(N)) 

#ci_mu_95 = stats.t.interval(gamma,N-1,loc=X_quer,scale=s_X_square/np.sqrt(N)) # Alternativ

c11 = stats.chi2.ppf((1-gamma)/2,N-1)
c22 = stats.chi2.ppf((1+gamma)/2,N-1)

ci_sigma_95 = (s_X_square*(N-1)/c2,s_X_square*(N-1)/c1)