# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:01:37 2020

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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
import pandas as pd

# Daten als Pandas Dataframe importieren
data = loadmat('Glasuntersuchung')['values']
df = pd.DataFrame(data, columns=['n', 'rho/(g/cm^3)'])

# Streudiagramm
ax1 = df.plot.scatter(x='rho/(g/cm^3)',y='n')
ax1.set_title('Scatterplot')

# kumulative Randhäufigkeiten
n_sort = df['n'].sort_values()
n_sort = np.append(0,n_sort)

rho_sort = df['rho/(g/cm^3)'].sort_values()
rho_sort = np.append(0,rho_sort)

H = np.array([np.cumsum(1/len(n_sort)*np.ones(n_sort.size-1))])
H = np.append(0,H)

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.step(n_sort, H)
ax1.set_title('Cumsum n')
ax2.step(rho_sort, H)
ax2.set_title(r'Cumsum $\rho$')


# Charakterisieren der Stichprobe
n_quer, roh_quer = df.mean(axis=0)
s_n, s_rho = df.std(axis=0)
s_nrho = df.cov()
#r_nrho = df.corr()