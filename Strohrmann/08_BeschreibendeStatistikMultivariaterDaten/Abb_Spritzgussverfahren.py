# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:58:55 2020

@author: stma0003
"""

""" Bibliotheken importieren"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Laden der Daten und Initialisiseren der Variablen"""
values= loadmat('spritzguss')
d = values['d']
T = values['T']
p = values['p']/1000
S = values['s']
X = np.append(np.append(np.append(d,T,axis=0),p,axis=0),S,axis=0)

"""Kumulative Randhäufigkeiten berechnen"""
dsort = np.append(np.append(0.0,np.sort(d)),10)
Tsort = np.append(np.append(20,np.sort(T)),130)
psort = np.append(np.append(0.4,np.sort(p)), 1.1)
Ssort = np.append(np.append(0.5,np.sort(S)), 3)
H =np.append([0,0],np.cumsum(1/len(T)*np.ones(np.shape(T))))

"""Kumulative Randhäufigkeiten darstellen"""
f1 = plt.figure(1, figsize=(12, 9))
axes1 = f1.subplots(2,2,gridspec_kw=dict(hspace=0.3))

axes1[0,0].step(dsort,H, color='b')
axes1[0,0].grid(True, ls='--')
axes1[0,0].set_xlabel('Wanddicke d / mm')
axes1[0,0].set_ylabel('Kumulative Randhäufigkeit H$_d$(d)')
axes1[0,0].set_title('Wanddicke')

axes1[0,1].step(Tsort, H,color='b')
axes1[0,1].grid(True, ls='--')
axes1[0,1].set_xlabel('Temperatur T / °C')
axes1[0,1].set_ylabel('Kumulative Randhäufigkeit H$_T$(T)')
axes1[0,1].set_title('Temperatur')

axes1[1,0].step(psort, H,color='b')
axes1[1,0].grid(True, ls='--')
axes1[1,0].set_xlabel('Nachdruck p / kbar')
axes1[1,0].set_ylabel('Kumulative Randhäufigkeit H$_p$(p)')
axes1[1,0].set_title('Nachdruck')

axes1[1,1].step(Ssort, H,color='b')
axes1[1,1].grid(True, ls='--')
axes1[1,1].set_xlabel('Schwindung S / %')
axes1[1,1].set_ylabel('Kumulative Randhäufigkeit H$_S$(S)')
axes1[1,1].set_title('Schwindung')

"""Streudiagramm als Matrix"""
df = pd.DataFrame(np.transpose(X), columns=['d / mm', 'T / °C', 'p / kbar', 'S / %'])
axes2 = pd.plotting.scatter_matrix(df, alpha=1,figsize=(12, 8), Color='b', hist_kwds=dict(Color='b'))

""" Kennwerte berechnen in Numpy """
mX = np.mean(X, axis=1)
cX = np.cov(X)

""" Kennwerte berechnen in Pandas """
mX2 = df.mean()
cX2 = df.cov()



