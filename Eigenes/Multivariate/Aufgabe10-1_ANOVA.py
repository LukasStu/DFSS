# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:00:00 2020

@author: Lukas Stürmlinger Matrikel-Nummer: 73343
"""
""" Waschmitteltest: Hat das Waschmittel einen signifikaten Einfluss auf die Reinheit?"""

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
import statsmodels.api as sm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
from statsmodels.formula.api import ols

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('Waschmitteltest.mat')['Testergebnis']

df = pd.DataFrame({'Waschmittel':np.repeat(['A', 'B', 'C', 'D'], 3),
                   'Reinheit': data.reshape(-1)})

""" ANOVA durchführen, dazu Modell aufbauen
    C(...) sind kategorische Variablen
    C(...):C(...) ist das Produkt zweier kategorischer Variablen
    type=2 ist wieder ein dataframe """
model = ols('Reinheit ~ C(Waschmittel)', data=df).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
pVal = anova1.loc['C(Waschmittel)','PR(>F)']

print('p-Value = {:.3f}'.format(pVal))

# Bewertung des p-Values
if pVal > 0.05:
    print('-> Die Reinheit hängt nicht vom Waschmittel ab\n')
else:
    print('-> Die Reinheit hängt vom Waschmittel ab\n')

df2 = pd.DataFrame({'Waschmaschine':np.repeat(['1', '2', '3'], 4),
                    'Reinheit': data.T.reshape(-1)})

""" ANOVA durchführen, dazu Modell aufbauen
    C(...) sind kategorische Variablen
    C(...):C(...) ist das Produkt zweier kategorischer Variablen
    type=2 ist wieder ein dataframe """
model2 = ols('Reinheit ~ C(Waschmaschine)', data=df2).fit()
anova2 = sm.stats.anova_lm(model2, typ=2)
pVal2 = anova2.loc['C(Waschmaschine)','PR(>F)']

print('p-Value = {:.3f}'.format(pVal2))

# Bewertung des p-Values
if pVal2 > 0.05:
    print('-> Die Reinheit hängt nicht von der Waschmaschine ab\n')
else:
    print('-> Die Reinheit hängt von der Waschmaschine ab\n')
