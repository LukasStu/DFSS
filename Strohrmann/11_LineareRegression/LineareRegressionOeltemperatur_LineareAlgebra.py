# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:10:34 2020

@author: abka0001
"""

from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence


""" Konfidenzbereich und Prognosebereich bei Regressionsfunktionen """
""" Auswahl beliebiger Stützstellen in regress_ex Dataframe """
""" Herleitung in Vorlesungsunterlagen DFSS """

def conf_pred_band_ex(regress_ex , poly, model, alpha=0.05):
    
    # Regressionsfunktion übernehmen und Matrix der neuen Stützstellen aufbauen
    # Neue Stützstellen übernehmen, alle Terme entsprechen einer Spalte 
    poly_ex = ols(poly.formula,regress_ex)    
    x0 = poly_ex.exog
    
    # Konfidenz- und Prognodebereich berechnen
    # Kenngrößen aus model verwenden, weil das zur ursprünglichen Regression verwendet wurde
    infl = OLSInfluence(model)    
    d = np.dot(x0,np.dot(infl.results.model.normalized_cov_params,x0.T))
    tppf = stats.t.isf(alpha/2, model.df_resid)
    lconf_ex = tppf*np.sqrt(np.diag(d)*model.mse_resid)
    lprog_ex=tppf *np.sqrt((1+np.diag(d))*model.mse_resid)
    
    return lconf_ex, lprog_ex


""" Messdaten aus Beispiel übernehmen und grafisch analysieren """

regress = pd.DataFrame({'Temperatur': np.arange(0,110,10),
                        'Spannung': [2.7660, 2.8626, 3.0054, 3.1202, 3.1738, 3.4114, 
                                     3.6761, 3.8033, 3.9440, 4.1887, 4.1659]})
fig1 = plt.figure(1, figsize=(12, 4))
fig1.suptitle('')
ax1, ax2 = fig1.subplots(1,2)
ax1.axis([-10,110,2,5])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Spannung $U$ / V');  
ax1.set_title('Lineare Regression');  
ax1.grid(True)


""" Lineares Regressionsmodell definieren und berechnen """

poly = ols("Spannung ~ Temperatur + I(Temperatur**2)+I(Temperatur**3)" , regress)
model = poly.fit()
print(model.summary())


""" Konfidenz- und Prognosebereich an den bekannten Stützstellen berechnen """

st, data, ss2 = summary_table(model, alpha=0.05)
regress['Fit'] = data[:, 2]
regress['Conll'] = data[:, 4]
regress['Conul'] = data[:, 5]
regress['Predll'] = data[:, 6]
regress['Predul'] = data[:, 7]


""" Konfidenzbereich und Prognosebereich bei Regressionsfunktionen """
""" Vergleich unterschiedlicher Verfahren """

# Berechnung mit Herleitung aus der Vorlesung 
lconf,lprog = conf_pred_band_ex(regress, poly, model, alpha=0.05)
LL_Konf, Ul_Konf = model.predict(regress['Temperatur']) - lconf, model.predict(regress['Temperatur']) + lconf
LL_Pred, Ul_Pred = model.predict(regress['Temperatur']) - lprog, model.predict(regress['Temperatur']) + lprog
ax1.plot(regress['Temperatur'],LL_Konf,'b--',label='LL Konf_Manuell')
ax1.plot(regress['Temperatur'],Ul_Konf,'b--',label='UL Konf_Manuell')
ax1.plot(regress['Temperatur'],LL_Pred,'b',label='LL Pred_Manuell')
ax1.plot(regress['Temperatur'],Ul_Pred,'b',label='LL Pred_Manuell')

# Werte aus summary table
ax1.plot(regress['Temperatur'],regress['Spannung'],'bo', label='Daten')
ax1.plot(regress['Temperatur'],regress['Fit'],'b', label='lineare Regression')
ax1.plot(regress['Temperatur'],regress['Conll'],'r--',label='LL Konf_Statsmodel')
ax1.plot(regress['Temperatur'],regress['Conul'],'r--',label='UL Konf_Statsmodel')
ax1.plot(regress['Temperatur'],regress['Predll'],'g',label='LL Pred_Statsmodel')
ax1.plot(regress['Temperatur'],regress['Predul'],'g',label='LL Pred_Statsmodel')
ax1.legend()


""" Berechnung und Darstellung der Residuen """

ax2.stem(regress['Temperatur'], model.resid, 'r', use_line_collection=True, markerfmt='ro')
ax2.axis([-10,110,-0.2,0.2])
ax2.set_xlabel('Temperatur $T$ / °C');
ax2.set_ylabel('Abweichung Spannung $\u0394U$ / V');  
ax2.set_title('Residuen');  
ax2.grid(True)


""" Berechnung von Kondidenz- und Prognosebereich für beliebige Stützstellen""" 

# Dataframe aufbauen und mit Regressionswerten befüllen
regress_ex = pd.DataFrame({'Temperatur': np.arange(-50,150,5)})
regress_ex['Spannung'] = model.predict(regress_ex['Temperatur'])

# Berechnung mit Herleitung aus der Vorlesung 
lconf_ex,lprog_ex = conf_pred_band_ex(regress_ex , poly, model, alpha=0.05)
LL_Konf, Ul_Konf = model.predict(regress_ex['Temperatur'])-lconf_ex, model.predict(regress_ex['Temperatur'])+lconf_ex
LL_Pred, Ul_Pred = model.predict(regress_ex['Temperatur'])-lprog_ex, model.predict(regress_ex['Temperatur'])+lprog_ex


"""  Grafischer Vergleich """

fig2 = plt.figure(2, figsize=(6, 4))
fig2.suptitle('')
ax1= fig2.subplots(1,1)
ax1.axis([-50,150,2,5])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Spannung $U$ / V');  
ax1.set_title('Exterapolation der Konfidenzbereich und Prognosebereich ');  
ax1.grid(True)
ax1.plot(regress['Temperatur'],regress['Spannung'],'bo', label='Daten')
ax1.plot(regress_ex['Temperatur'],model.predict(regress_ex['Temperatur']),'b', label='lineare Regression')
ax1.plot(regress_ex['Temperatur'],LL_Konf,'r--',label='LL Konf_EX')
ax1.plot(regress_ex['Temperatur'],Ul_Konf,'r--',label='UL Konf_EX')
ax1.plot(regress_ex['Temperatur'],LL_Pred,'g',label='LL Pred_EX')
ax1.plot(regress_ex['Temperatur'],Ul_Pred,'g',label='UL Pred_EX')
ax1.legend()


""" Seaborn: Berechnung und Darstellung dem Konfidenzbereich sowie der Extrapolation """ 
""" Vorsicht: Ergebnisse stimmen nicht mit den anderen Daten überein """

fig3 = plt.figure(3, figsize=(12, 8))
fig3.suptitle('')
ax1= fig3.subplots(1,1)
ax1.plot(regress_ex['Temperatur'],LL_Konf,'r--',label='LL Konf_EX')
ax1.plot(regress_ex['Temperatur'],Ul_Konf,'r--',label='UL Konf_EX')
ax1.axis([-50,150,2,5])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Spannung $U$ / V');  
ax1.set_title('Seaborn_regplot');  
ax1.grid(True)

import seaborn as sns
ax1 = sns.regplot(x='Temperatur',y='Spannung', data=regress,ci=95,
                  color='blue',order=3, robust=False, truncate=False,
                  scatter_kws={'color': 'red','s': 100,'alpha': 0.9})

