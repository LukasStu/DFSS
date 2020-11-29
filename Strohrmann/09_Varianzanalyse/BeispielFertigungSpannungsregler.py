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
df = pd.DataFrame({'Schicht': np.tile(np.repeat([1, 2, 3], 3), 3),
                   'Linie':np.repeat(['A', 'B', 'C'], 9),
                   'Spannung': [16.1736, 16.0336, 16.0971,
                                16.1243, 15.9743, 16.0653,
                                15.9059, 15.8825, 15.8979,
                                16.4598, 16.5174, 16.4884,
                                16.7064, 16.5755, 16.4482,
                                16.7010, 16.7071, 16.7317,
                                16.4500, 16.5278, 16.3452,
                                16.5261, 16.4987, 16.4420,
                                16.5136, 16.2742, 16.1590]})

print('Datensatz')
print()
print(df)

""" ANOVA durchführen, dazu Modell aufbauen
    C(...) sind kategorische Variablen
    C(...):C(...) ist das Produkt zweier kategorischer Variablen
    type=2 ist wieder ein dataframe """
model = ols('Spannung ~ C(Linie) + C(Schicht) + C(Linie):C(Schicht)', data=df).fit()
anova2 = sm.stats.anova_lm(model, typ=2)

print()
print()
print('ANOVA-Tabelle')
print()
print(anova2)

""" Boxplot erstellen """
fig = plt.figure(2, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)
ax1 = df.boxplot('Spannung',by='Linie',ax = ax1)
ax1.axis([0.5, 3.5, 15.8, 16.8])
ax1.set_xlabel('Linie');
ax1.set_ylabel('Spannung U / V');  
ax1.set_title('');  
ax1.grid(True, which='both', axis='both', linestyle='--')

ax2 = df.boxplot('Spannung',by='Schicht',ax=ax2)
ax2.axis([0.5, 3.5, 15.8, 16.8])
ax2.set_xlabel('Schicht');
ax2.set_ylabel('Spannung U / V');  
ax2.set_title('');  
ax2.grid(True, which='both', axis='both', linestyle='--')

plt.suptitle('')
