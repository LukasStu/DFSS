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
"""m) hier muss ein Tippfehler sein. Deshalb ist es auskommentiert"""
"""Bitte bewerten"""
"""Außerdem scheine ich mit den Grafikvariablen durcheinander gekommen zu sein..."""
# Assessment of process variation according to prodedure 3
Y_TOLERANCE = 0.01
Y_REPEAT_REFERENCE = 1.5
#Load, format and evaluate data
data = loadmat('Streuverhalten')
y_variation_3 = pd.DataFrame({'Part': np.tile(np.arange(0, 25, 1), 2),
                              'Measurement': np.repeat([1, 2], 25),
                              'Value': data['h'].reshape(-1, order='F')})
Y_K = 25
Y_N = 2


# Calculation of normalized squares sums making use of anova table
model = ols('Value ~ C(Part)', data=y_variation_3).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
anova1["M"] = anova1["sum_sq"]/anova1["df"]

# estimations of variance and calculation of GRR and ndc
equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
part_variation = np.sqrt((anova1.loc["C(Part)", "M"]
                          - anova1.loc["Residual", "M"])/Y_N)
grr = equipment_variation
grr_relative = 6*grr/Y_TOLERANCE
ndc = 1.41*part_variation/grr
print("")
print("")
print("Streuverhalten: Verfahren 3")
print("")
print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
print("Number of Distict Categories ndc = ", round(ndc, 3))

# Visualization
y_variation_3_multi\
    = y_variation_3.set_index(['Measurement', 'Part'])
fig6 = plt.figure(4, figsize=(12, 4))
fig6.suptitle('')
ax1, ax2 = fig6.subplots(1, 2)
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],
          'b', label='Messung 1')
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],
          'r:', label='Messung 2')
#ax1.axis([0, 26, 0.9, 1.1])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Höhe/m')
ax1.set_title('Streuverhalten')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)




