# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:51:37 2021

@author: LStue
"""

from scipy import stats
import scipy.io
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm


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
def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], scipy.io.matlab.mio5_params.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

# Assesment of Resolution

Y_TOLERANCE = 0.1
Y_RESOLUTION = 0.001
print("")
print("")
print("1. Bewertung der Auflösung")
print("")
if Y_RESOLUTION/Y_TOLERANCE <= 0.05:
    print("Auflösung ausreichend")
else:
    print("Auflösung ist nicht ausreichend")
    
    
# Systematic Deviation and Repeatability

# Load data set and reference
data = loadmat('MSATemperatur')
y_repeat_test = data["Temperaturmessung"]["Verfahren1"]["data"]
np.append(y_repeat_test, data["Temperaturmessung"]["Verfahren2"]["data"])
np.append(y_repeat_test, data["Temperaturmessung"]["Verfahren4"]["data"])
np.append(y_repeat_test, data["Temperaturmessung"]["Verfahren5"]["data"])
y_repeat_len = np.size(y_repeat_test)
Y_REPEAT_REFERENCE = 18.3

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(np.arange(0, y_repeat_len)+1, y_repeat_test, 'bo-')
ax1.plot(np.arange(0, y_repeat_len)+1, Y_REPEAT_REFERENCE*np.ones(y_repeat_len), 'r')
ax1.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE+0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
ax1.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE-0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
#ax1.axis([0, 26, 4.994, 5.006])
ax1.set_xlabel('Messung')
ax1.set_ylabel('Temperatur $T$ / $^\circ$C')
ax1.set_title('Visualisierung der systematischen Messabweichung')
ax1.grid(True)

# Calculation of capability index
y_deviation = np.mean(y_repeat_test) - Y_REPEAT_REFERENCE
c_g = 0.1*Y_TOLERANCE/3/np.std(y_repeat_test, ddof=1)
print("")
print("")
print("2. Systematische Abweichung und Wiederholbarkeit")
print("")
print("C_g = ", round(c_g, 3))
if c_g >= 1.33:
    print("Wiederholbarkeit ausreichend")
else:
    print("Wiederholbarkeit ist nicht ausreichend")
    
c_gk = (0.1*Y_TOLERANCE - np.abs(y_deviation))/3/np.std(y_repeat_test, ddof=1)
print("")
print("C_gk = ", round(c_gk, 3))
if c_gk >= 1.33:
    print("Wiederholbarkeit und sytematische Abweichung ausreichend")
elif c_g >= 1.33:
    print("Systematische Abweichung zu groß")
else:
    print("Auflösung und systematische Abweichung nicht ausreichend")
    
# Hypothesistest with H0: y_repeat_test = Y_REPEAT_REFERENCE
hypo_test = stats.ttest_1samp(y_repeat_test, Y_REPEAT_REFERENCE)
print("")
print("Hypothesentest auf Abweichung mit p-value = ",
      round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")