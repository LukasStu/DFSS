# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:03:41 2020

@author: stma0003
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import rlm

""" Messdaten aus Beispiel übernehmen und grafisch analysieren """

#regress = pd.DataFrame({'Temperatur': np.arange(0,110,10),
#                        'Spannung': [2.7660, 2.8626, 3.0054, 3.1202, 3.1738, 3.4114,
#                                      3.6761, 3.8033, 2.9440, 4.1887, 4.1659]})

regress_a = pd.DataFrame({'Temperatur': np.arange(0,110,10),
                         'Spannung': [2.7660, 2.8626, 3.0054, 3.1202, 3.1738, 3.4114,
                                      3.6761, 3.8033, 2.9440, 4.1887, 4.1659]})


regress_b = pd.DataFrame({'Temperatur': np.arange(0,110,10),
                         'Spannung': [2.7660, 2.8626, 3.0054, 3.1202, 3.1738, 3.4114,
                                      3.6761, 3.8033, 3.9440, 4.1887, 4.1659]})



""" Lineares Regressionsmodell definieren und berechnen """

model_a = ols("Spannung ~ Temperatur " , regress_a).fit()
print(model_a.summary())
st, data_a, ss2 = summary_table(model_a, alpha=0.05)

""" Darstellung der Regressionsfunktion zusammen mit Originaldaten """

regress_a['Fit'] = data_a[:, 2]



model_b = ols("Spannung ~ Temperatur " , regress_b).fit()
print(model_b.summary())
st, data_b, ss2 = summary_table(model_b, alpha=0.05)

""" Darstellung der Regressionsfunktion zusammen mit Originaldaten """

regress_b['Fit'] = data_b[:, 2]





fig = plt.figure(1, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)

ax1.plot(regress_a['Temperatur'],regress_a['Spannung'],'ro')
ax1.plot(regress_a['Temperatur'],regress_a['Fit'],'r--')

ax1.plot(regress_b['Temperatur'],regress_b['Spannung'],'bo')
ax1.plot(regress_b['Temperatur'],regress_b['Fit'],'b--')

ax1.axis([-10,110,2,5])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Spannung $U$ / V');  
ax1.set_title('Robuste Regression');  
ax1.grid(True)


""" Lineares Robuste Regression definieren und berechnen """

model = rlm("Spannung ~ Temperatur", regress_a, M=sm.robust.norms.AndrewWave()).fit()
print(model.summary())
regress_a['Fit_robust'] = model.fittedvalues
regress_a['Gewichtungsfaktor'] = model.weights

ax1.plot(regress_a['Temperatur'],regress_a['Fit_robust'],'g')


ax2.bar(regress_a['Temperatur'],regress_a['Gewichtungsfaktor'],5, color='b' )
ax2.set_xlabel('Temperatur $T$ / °C');
ax2.grid(True)
ax2.set_ylabel('Gewichtungsfaktor');


