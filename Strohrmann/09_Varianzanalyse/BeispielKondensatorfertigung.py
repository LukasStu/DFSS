# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:41:19 2020

@author: stma0003
"""
    
""" Bibliotheken importieren"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

""" Data Frame Variable mit Daten erstellen """
df = pd.DataFrame({'Linie': np.repeat([1, 2, 3], 4),
                   'Kapazitaet': [46.60,48.20,43.74,46.60,
                                 47.56,47.24,40.24,46.60,
                                 48.20,47.56,45.30,47.88]})
print()
print('Datensatz')
print()
print(df)

""" ANOVA durchführen """
model = ols('Kapazitaet ~ C(Linie)', data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)

print()
print()
print('ANOVA-Tabelle')
print()
print(anova)

