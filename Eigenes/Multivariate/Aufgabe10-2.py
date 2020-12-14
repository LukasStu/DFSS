# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:43:47 2020

@author: lukas
"""

"""  Initialisierung: Variablen löschen, Konsole leeren """    
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

""" Bibliotheken importieren"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
from statsmodels.formula.api import ols

""" G-Lebensdauer abhängig vom Material und Temperatur?"""

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""

data1 = loadmat('LebensdauerGeneratoren.mat')['Data1']
# Data Frame Variable mit Daten erstellen
data1 = data1.reshape(-1)
df1 = pd.DataFrame(data1)
df1.insert(0,'Material',np.repeat([1,2,3],8))
df1.insert(1,'Temperatur',np.tile(['T10','T18'],12))
df1 = df1.rename(columns = {0:'Lebensdauer'})

""" ANOVA durchführen, dazu Modell aufbauen
    C(...) sind kategorische Variablen
    C(...):C(...) ist das Produkt zweier kategorischer Variablen
    type=2 ist wieder ein dataframe """
model1 = ols('Lebensdauer ~ C(Material) + C(Temperatur) + C(Material):C(Temperatur)', data=df1).fit()
anova1 = sm.stats.anova_lm(model1, typ=2)

print(anova1)

""" p-Values bewerten """
pValTemp = anova1.loc['C(Temperatur)','PR(>F)']
pValMat = anova1.loc['C(Material)','PR(>F)']

print('p-Value Tempertur = {:.3f}\n'.format(pValTemp))

# Bewertung des p-Values
if pValTemp > 0.05:
    print('-> Die Lebensdauer hängt nicht von der Temperatur ab.\n')
else:
    print('-> Die Lebensdauer hängt von der Temperatur ab.\n')

print('p-Value Material = {:.3f}\n'.format(pValMat))

# Bewertung des p-Values
if pValMat > 0.05:
    print('-> Die Lebensdauer hängt nicht vom Material ab.\n')
else:
    print('-> Die Lebensdauer hängt nicht vom Material ab.\n')

""" Boxplot erstellen """
fig = plt.figure(2, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)
ax1 = df1.boxplot('Lebensdauer',by='Temperatur',ax = ax1)
#ax1.axis([0.5, 3.5, 15.8, 16.8])
ax1.set_xlabel('Temperatur');
ax1.set_ylabel('Lebensdauer');  
ax1.set_title('');  
ax1.grid(True, which='both', axis='both', linestyle='--')

ax2 = df1.boxplot('Lebensdauer',by='Material',ax=ax2)
#ax2.axis([0.5, 3.5, 15.8, 16.8])
ax2.set_xlabel('Lebensdauer');
ax2.set_ylabel('Material');  
ax2.set_title('');  
ax2.grid(True, which='both', axis='both', linestyle='--')

plt.suptitle('')



