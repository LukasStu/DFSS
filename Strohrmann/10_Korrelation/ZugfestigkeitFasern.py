# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""


""" Bibliotheken importieren"""
import pandas as pd
import numpy as np
import scipy.stats  as stats
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import t, norm, chi2, f
import statsmodels.api as sm
from statsmodels.formula.api import ols

""" Korrelation von Trocknungszeit und Zugfestigkeit von Fasern """
""" Stichprobenwerte aus Aufgabe übernehmen und als Dataframe speichern """

data = loadmat('ZugfestigkeitFasern')
df = pd.DataFrame({'Trocknungszeit': data['values'][:,1],
                   'Zugfestigkeit': data['values'][:,2]})

print(' ')
print(df)

""" Grafische Darstellung """

fig = plt.figure(1, figsize=(6, 4))
fig.suptitle('')
ax1 = fig.subplots(1,1)
ax1.plot(df['Trocknungszeit'],df['Zugfestigkeit'],'bo')
ax1.axis([20,45,210,250])
ax1.set_xlabel('Trocknungszeit t / h');
ax1.set_ylabel('Normierte Zugfestigkeit R ');  
ax1.set_title('');  
ax1.grid(True, which='both', axis='both', linestyle='--')

""" Berechnung Korrelation über Dataframe """

Corr1 = round((df.corr(method='pearson')).loc['Trocknungszeit','Zugfestigkeit'],3)

""" Berechnung Korrelation und Hypothesentest roh = 0 über scipy.stats """

Corr2, p1 = stats.pearsonr(df['Trocknungszeit'],df['Zugfestigkeit'])
Corr2 = round(Corr2,3)

print(' ')
print('Korrelation zwischen den Größen (df) : ', Corr1)
print('Korrelation zwischen den Größen (scipy) : ', Corr2)
print('p-value zur Korrelation : ', round(p1,5))

""" Berechnung des Konfidenzbereichs """
N = df.size
Corr = df.corr(method='pearson').loc['Trocknungszeit','Zugfestigkeit']
gamma = 0.95
c1 = norm.ppf((1-gamma)/2)
c2 = norm.ppf((1+gamma)/2)
roh1 = round(np.tanh(np.arctanh(Corr)-c2/np.sqrt(N-3)),3)
roh2 = round(np.tanh(np.arctanh(Corr)-c1/np.sqrt(N-3)),3)

print(' ')
print('Untere Grenze des Konfidenzbereichs: ', roh1)
print('Obere Grenze des Konffidenzbereichs: ', roh2)


model = ols('Zugfestigkeit ~ Trocknungszeit', data=df).fit()
print(model.summary())


