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

""" Messdaten aus Beispiel übernehmen und grafisch analysieren """

regress = pd.DataFrame({'Temperatur': np.arange(0,110,10),
                        'Spannung': [2.7660, 2.8626, 3.0054, 3.1202, 3.1738, 3.4114, 
                                     3.6761, 3.8033, 3.9440, 4.1887, 4.1659]})
fig = plt.figure(1, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(regress['Temperatur'],regress['Spannung'],'bo')
ax1.axis([0,100,2.5,4.5])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Spannung $U$ / V');  
ax1.set_title('Lineare Regression');  
ax1.grid(True)

""" Lineares Regressionsmodell definieren und berechnen """

# model = ols("Spannung ~ Temperatur", regress).fit()
# model = ols("Spannung ~ Temperatur + I(Temperatur**2) + I(Temperatur**3)" , regress).fit()
model = ols("Spannung ~ I(Temperatur**2) + I(Temperatur**3)" , regress).fit()
print(model.summary())
st, data, ss2 = summary_table(model, alpha=0.05)

""" Darstellung der Regressionsfunktion zusammen mit Originaldaten """

regress['Fit'] = data[:, 2]
# regress['Resid'] = data[:,3]
ax1.plot(regress['Temperatur'],regress['Fit'],'b')

""" Berechnung und Darstellung der Residuen """

ax2.stem(regress['Temperatur'], model.resid, 'r', use_line_collection=True, markerfmt='ro')
ax2.axis([0,100,-0.2,0.2])
ax2.set_xlabel('Temperatur $T$ / °C');
ax2.set_ylabel('Abweichung Spannung $\u0394U$ / V');  
ax2.set_title('Residuen');  
ax2.grid(True)

""" Konfidenzbereich und Prognosebereich bei Regressionsfunktionen """
regress['Fit'] = data[:, 3]
regress['Conll'] = data[:, 4].T
regress['Conul'] = data[:, 5].T
regress['Predll'] = data[:, 6].T
regress['Predul'] = data[:, 7].T

ax1.plot(regress['Temperatur'],regress['Conll'],'r--')
ax1.plot(regress['Temperatur'],regress['Conul'],'r--')
ax1.plot(regress['Temperatur'],regress['Predll'],'g')
ax1.plot(regress['Temperatur'],regress['Predul'],'g')

