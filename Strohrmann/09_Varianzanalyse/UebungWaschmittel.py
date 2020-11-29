# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:41:19 2020

@author: stma0003
"""

   
""" Bibliotheken importieren"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

""" Data Frame Variable mit Daten erstellen """
df = pd.DataFrame({'Waschmittel':np.repeat(['A', 'B', 'C', 'D'], 3),
                   'Reinheit': [53, 50, 59,
                                54, 54, 60,
                                56, 58, 62,
                                50, 45, 57]})

print('Datensatz')
print()
print(df)

""" ANOVA durchführen, dazu Modell aufbauen
    C(...) sind kategorische Variablen
    C(...):C(...) ist das Produkt zweier kategorischer Variablen
    type=2 ist wieder ein dataframe """
model = ols('Reinheit ~ C(Waschmittel)', data=df).fit()
anova1 = sm.stats.anova_lm(model, typ=2)

print()
print()
print('ANOVA-Tabelle')
print()
print(anova1)

""" Boxplot erstellen """
fig = plt.figure(2, figsize=(6, 4))
fig.suptitle('')
ax1 = fig.subplots(1,1)
ax1 = df.boxplot('Reinheit',by='Waschmittel',ax = ax1)
ax1.axis([0.5, 4.5, 44, 64])
ax1.set_xlabel('Waschmittel');
ax1.set_ylabel('Reinheit');  
ax1.set_title('');  
ax1.grid(True, which='both', axis='both', linestyle='--')


# plt.suptitle('')
