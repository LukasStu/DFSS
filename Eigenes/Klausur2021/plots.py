

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:00:00 2020

@author: Lukas Stürmlinger Matrikel-Nummer: 73343
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
from scipy import stats
from scipy.stats import t, norm, uniform, chi2, f
from scipy.io import loadmat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.formula.api import ols
import sympy as syms
from sympy.core import evalf

def conf_pred_band_ex(_regress_ex, _poly, _model, alpha=0.05):
    """ Function calculates the confidence and prediction interval for a
    given multivariate regression function poly according to lecture DFSS,
    regression parameters are already determined in an existing model,
    identical polynom is used for extrapolation

    Parameters
    ----------
    regress_ex : DataFrame
        Extended dataset for calculation.
    poly : OLS object of statsmodels.regression.linear_model modul
        definition of regression model.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Model parameters.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    lconf_ex : Series
        Distance of confidence limit to mean regression.
    lpred_ex : Series
        Distance of prediction limit to mean regression.
    """

    # ols is used to calculte the complets vector x_0 of input variables
    poly_ex = ols(_poly.formula, _regress_ex)
    x_0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x_0, np.dot(_poly.normalized_cov_params, x_0.T))
    c_1 = stats.t.isf(alpha/2, _model.df_resid)
    lconf_ex = c_1*np.sqrt(np.diag(d)*_model.mse_resid)
    lpred_ex = c_1*np.sqrt((1+np.diag(d))*_model.mse_resid)

    return lconf_ex, lpred_ex

"""a) Histogramme """
data = loadmat('Signifikanz')['signifikanz']
data10 = data[:,0].reshape(-1)
data20 = data[:,1].reshape(-1)

#absolute Häufigkeit
fig1, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(data10)
ax1.set_xlabel(r'c/ m/s bei 10°C')
ax1.set_ylabel(r'Absolute Häufigkeit')
ax2.hist(data20)
ax2.set_xlabel(r'c/ m/s bei 20°C')
ax2.set_ylabel(r'Absolute Häufigkeit')



"""b) Parameter schätzen""" 
c10_quer = np.mean(data10)
c20_quer = np.mean(data20)
s_c10 = np.std(data10,ddof=1)
s_c20 = np.std(data20,ddof=1)
N10 = np.size(data10)
N20 = np.size(data20)


# # Wahrscheinlichkeitsverteilung
fig2, (ax3, ax4) = plt.subplots(1,2)
ax3.hist(data10, density='true')
ax3.set_xlabel(r'c/ m/s bei 10°C')
ax3.set_ylabel(r'Wahrscheinlichkeit')
ax4.hist(data20, density='true')
ax4.set_xlabel(r'c/ m/s bei 10°C')
ax4.set_ylabel(r'Wahrscheinlichkeit')

# # Dichtefunktion plotten
xaxes10 = np.arange(min(data10), max(data10) ,1E-3)
xaxes20 = np.arange(min(data20), max(data20) ,1E-3)
f10 = norm.pdf(xaxes10, loc=c10_quer, scale=s_c10)
f20 = norm.pdf(xaxes20, loc=c20_quer, scale=s_c20)
ax3.plot(xaxes10, f10, 'r')
ax4.plot(xaxes20, f20, 'r')

# Linearity
Y_TOLERANCE = 0.01
Y_REPEAT_REFERENCE = 1.5
# Load data set and reference
data = loadmat('Linearitaet')['linearitaet']
y_linearity = pd.DataFrame({'reference': np.repeat([1.1, 1.3,
                                                    1.5, 1.7, 1.9], 10),
                            'value': data.reshape(-1, order='F')})
y_linearity["deviation"] = y_linearity["value"] - y_linearity["reference"]

# Visualization
fig = plt.figure(0, figsize=(12, 4))
fig.suptitle('')
ax6, ax7 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
ax6.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax1.axis([0, 60, -0.1, 0.1])
ax6.set_xlabel('Referenzwert $T$ / °C')
ax6.set_ylabel(r' Abweichung $\Delta T$ / °C')
ax6.set_title('Bewertung des Konfidenzbereichs')
ax6.grid(True)
ax7.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax2.axis([0, 60, -0.1, 0.1])
ax7.set_xlabel('Referenzwert $T$ / °C')
ax7.set_ylabel(r' Abweichung $\Delta T$ / °C')
ax7.set_title('Mittelwerte zur Lineartätsbewertung')
ax7.grid(True)

# Regression function with confidence bounds
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
print(model.summary())
y_plot = np.arange(0, 2, 1E-3)
y_regress = pd.DataFrame({"reference": np.reshape(y_plot, -1)})
y_regress["deviation"] = model.predict(y_regress)
ax6.plot(y_regress["reference"], y_regress["deviation"], 'r')
y_regress["confidence"], y_regress["prediction"] = \
    conf_pred_band_ex(y_regress, poly, model)
ax6.plot(y_regress["reference"],
         y_regress["deviation"]+y_regress["confidence"], 'r:')
ax6.plot(y_regress["reference"],
         y_regress["deviation"]-y_regress["confidence"], 'r:')
print("")
print("")
print("3. Linearität")
print("")
print("Prüfung Regressionsgerade")
if (model.pvalues > 0.05).all(axis=None):
    print("Keine signifikante Abweichung zur Linearität")
else:
    print("Signifikante Abweichung zur Linearität")

# Position of mean values for each reference
ax7.plot(y_linearity.groupby("reference").aggregate('mean'), 'ro')
ax7.plot(y_linearity["reference"],
         -np.ones(np.size(y_linearity["reference"]))*Y_TOLERANCE*0.05, 'g--')
ax7.plot(y_linearity["reference"],
         np.ones(np.size(y_linearity["reference"]))*Y_TOLERANCE*0.05, 'g--')
print("")
print("Prüfung individueller Abweichungen")
if (np.abs(y_linearity.groupby("reference").aggregate("mean"))["deviation"]
        <= 0.05*Y_TOLERANCE).all(axis=None):
    print("Keine individuelle Abweichung zur Linearität")
else:
    print("Individuelle Abweichung zur Linearität")
    