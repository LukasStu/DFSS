# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:51:37 2021

@author: LStue
"""
"""  Initialisierung: Variablen löschen, Konsole leeren """    
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

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

Y_TOLERANCE = 1
Y_RESOLUTION = 0.0001
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
y_repeat_len = np.size(y_repeat_test)
Y_REPEAT_REFERENCE = 18.3

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(np.arange(0, y_repeat_len)+1, y_repeat_test, 'bo-',label="Messwerte")
ax1.plot(np.arange(0, y_repeat_len)+1, Y_REPEAT_REFERENCE*np.ones(y_repeat_len), 'r', label="Referzenwert")
ax1.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE+0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--', label="10%-Grenze")
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
    

# Datensatz für Linearitätsuntersuchung
y_linearity = pd.DataFrame({'reference': np.tile(np.reshape(data["Temperaturmessung"]["Verfahren4"]["ref"], -1),10),
                            'deviation': np.reshape(data["Temperaturmessung"]["Verfahren4"]["data"], -1, order='C')})
y_linearity['deviation'] = y_linearity['reference'] - y_linearity['deviation']
y_linearity.sort_values(by=['reference'], inplace=True)

# Visualization
fig2 = plt.figure(2, figsize=(12, 4))
fig2.suptitle('')
ax1, ax2 = fig2.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
ax1.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax1.axis([0, 10, -0.004, 0.004])
ax1.set_xlabel('Referenzwert')
ax1.set_xlabel('Referenzwert $T$ / $^\circ$C')
ax1.set_ylabel(r' Abweichung $\Delta T$ / $^\circ$C')
ax1.set_title('Bewertung des Konfidenzbereichs')
ax1.grid(True)
ax2.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax2.axis([0, 10, -0.004, 0.004])
ax2.set_xlabel('Referenzwert $T$ / $^\circ$C')
ax2.set_ylabel(r' Abweichung $\Delta T$ / $^\circ$C')
ax2.set_title('Mittelwerte zur Lineartätsbewertung')
ax2.grid(True)

# Regression function with confidence bounds
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
print(model.summary())
y_plot = np.arange(1, 55, 1)
y_regress = pd.DataFrame({"reference": np.reshape(y_plot, -1)})
y_regress["deviation"] = model.predict(y_regress)
ax1.plot(y_regress["reference"], y_regress["deviation"], 'r')
y_regress["confidence"], y_regress["prediction"] = \
    conf_pred_band_ex(y_regress, poly, model)
ax1.plot(y_regress["reference"],
         y_regress["deviation"]+y_regress["confidence"], 'r:')
ax1.plot(y_regress["reference"],
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
ax2.plot(y_linearity.groupby("reference").aggregate('mean'), 'ro')
ax2.plot(y_linearity["reference"],
         -np.ones(np.size(y_linearity["reference"]))*Y_TOLERANCE*0.05, 'g--')
ax2.plot(y_linearity["reference"],
         np.ones(np.size(y_linearity["reference"]))*Y_TOLERANCE*0.05, 'g--')
print("")
print("Prüfung individueller Abweichungen")
if (np.abs(y_linearity.groupby("reference").aggregate("mean"))["deviation"]
        <= 0.05*Y_TOLERANCE).all(axis=None):
    print("Keine individuelle Abweichung zur Linearität")
else:
    print("Individuelle Abweichung zur Linearität")
    
    

# Assessment of process variation according to prodedure 2

# Load, format and evaluate data
y_variation_2 = pd.DataFrame({'Part': np.tile(np.arange(0, 10, 1), 6),
                              'Measurement': np.tile(np.repeat([1, 2], 10), 3),
                              'Appraiser': np.repeat(['A', 'B', 'C'], 20),
                              'Value': data["Temperaturmessung"]["Verfahren2"]["data"].T.reshape(-1)})
Y_K = 10 # TODO: Enter number of objects
Y_J = 3 # TODO: Enter number of appraisers
Y_N = 2 # TODO: Enter number of measurements series

# Calculation of normalized squares sums making use of anova table
model = ols('Value ~ C(Part) + C(Appraiser) + C(Part):C(Appraiser)',
            data=y_variation_2).fit()
anova2 = sm.stats.anova_lm(model, typ=2)
anova2["M"] = anova2["sum_sq"]/anova2["df"]

# estimations of variance and calculation of GRR and ndc
equipment_variation = np.sqrt(anova2.loc["Residual", "M"])
appraiser_variation = np.sqrt((anova2.loc["C(Appraiser)", "M"]
                               - anova2.loc["C(Part):C(Appraiser)", "M"])
                              / Y_K / Y_N)
interaction_variation = np.sqrt((anova2.loc["C(Part):C(Appraiser)", "M"]
                                 - anova2.loc["Residual", "M"])/Y_N)
part_variation = np.sqrt((anova2.loc["C(Part)", "M"]
                          - anova2.loc["Residual", "M"])/Y_J/Y_N)
grr = np.sqrt(appraiser_variation**2 + interaction_variation**2
              + equipment_variation**2)
grr_relative = 6*grr/Y_TOLERANCE
ndc = 1.41*part_variation/grr
print("")
print("")
print("5. Streuverhalten: Verfahren 2")
print("")
print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
print("Number of Distict Categories ndc = ", round(ndc, 3))

# Visualization for each appraiser making use of multi index
y_variation_2_multi\
    = y_variation_2.set_index(['Appraiser', 'Measurement', 'Part'])
fig4 = plt.figure(4, figsize=(12, 4))
fig4.suptitle('')
ax1, ax2, ax3 = fig4.subplots(1, 3, gridspec_kw=dict(wspace=0.3))
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['A', 1, :],
         'b', label='Messung 1')
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['A', 2, :],
         'r:', label='Messung 2')
#ax1.axis([0, 11, 5.97, 6.04])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel('Durchmesser $D$ / mm')
ax1.set_title('Prüfer A')
ax1.grid(True)
ax2.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['B', 1, :],
         'b', label='Messung 1')
ax2.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['B', 2, :],
         'r:', label='Messung 2')
#ax2.axis([0, 11, 5.97, 6.04])
ax2.set_xlabel('Stichprobe')
ax2.set_ylabel('Durchmesser $D$ / mm')
ax2.set_title('Prüfer B')
ax2.grid(True)
ax3.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['C', 1, :],
         'b', label='Messung 1')
ax3.plot(np.arange(1, Y_K+1, 1), y_variation_2_multi.loc['C', 2, :],
         'r:', label='Messung 2')
#ax3.axis([0, 11, 5.97, 6.04])
ax3.set_xlabel('Stichprobe')
ax3.set_ylabel('Durchmesser $D$ / mm')
ax3.set_title('Prüfer C')
ax3.grid(True)
ax3.legend(loc=1, ncol=1)

# Visualization of mean for each appraiser making use of multi index
fig5 = plt.figure(5, figsize=(6, 4))
fig5.suptitle('')
ax1 = fig5.subplots(1, 1)
ax1.plot(np.arange(1, Y_K+1, 1),
         y_variation_2_multi.loc['A', :, :].mean(level=['Part']),
         'b', label='Prüfer A')
ax1.plot(np.arange(1, Y_K+1, 1),
         y_variation_2_multi.loc['B', :, :].mean(level=['Part']),
         'r:', label='Prüfer B')
ax1.plot(np.arange(1, Y_K+1, 1),
         y_variation_2_multi.loc['C', :, :].mean(level=['Part']),
         'g--', label='Prüfer C')
#ax1.axis([0, 11, 5.97, 6.04])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel('Durchmesser $D$ / mm')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)


# Long term stability

# Load and evaluate data
y_longterm = data["Temperaturmessung"]["Verfahren5"]["data"].T
Y_LONGTERM_MU = 18.3
Y_LONGTERM_SIG = 0.01
y_longterm_mean = np.mean(y_longterm, axis=1)
y_longterm_std = np.std(y_longterm, ddof=1, axis=1)
y_longterm_len = y_longterm.shape[1]
GAMMA_WARN = 0.95
GAMMA_CORRECT = 0.99

# Hypothesistest for mean
c1_warn = stats.norm.ppf((1-GAMMA_WARN)/2)
c2_warn = stats.norm.ppf((1+GAMMA_WARN)/2)
c1_correct = stats.norm.ppf((1-GAMMA_CORRECT)/2)
c2_correct = stats.norm.ppf((1+GAMMA_CORRECT)/2)
y_longterm_mean_warn_1 = Y_LONGTERM_MU \
    + c1_warn*Y_LONGTERM_SIG/np.sqrt(y_longterm_len)
y_longterm_mean_warn_2 = Y_LONGTERM_MU \
    + c2_warn*Y_LONGTERM_SIG/np.sqrt(y_longterm_len)
y_longterm_mean_correct_1 = + Y_LONGTERM_MU \
    + c1_correct*Y_LONGTERM_SIG/np.sqrt(y_longterm_len)
y_longterm_mean_correct_2 = Y_LONGTERM_MU \
    + c2_correct*Y_LONGTERM_SIG/np.sqrt(y_longterm_len)
print("")
print("")
print("4. Langzeitstabilität")
print("")
print("Mittelwert")
print("")
if ((y_longterm_mean > y_longterm_mean_warn_1).all(axis=None)
        & (y_longterm_mean < y_longterm_mean_warn_2).all(axis=None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
if ((y_longterm_mean > y_longterm_mean_correct_1).all(axis=None)
        & (y_longterm_mean < y_longterm_mean_correct_2).all(axis=None)):
    print("Eingriffsgrenzen nicht überschritten")
else:
    print("Eingriffsgrenzen überschritten")

# Hypothesistest for standard deviation
c1_warn = stats.chi2.ppf((1-GAMMA_WARN)/2, y_longterm_len-1)
c2_warn = stats.chi2.ppf((1+GAMMA_WARN)/2, y_longterm_len-1)
c1_correct = stats.chi2.ppf((1-GAMMA_CORRECT)/2, y_longterm_len-1)
c2_correct = stats.chi2.ppf((1+GAMMA_CORRECT)/2, y_longterm_len-1)
y_longterm_sig_warn_1 = np.sqrt(c1_warn/(y_longterm_len-1))*Y_LONGTERM_SIG
y_longterm_sig_warn_2 = np.sqrt(c2_warn/(y_longterm_len-1))*Y_LONGTERM_SIG
y_longterm_sig_correct_1 = np.sqrt(c1_correct/(y_longterm_len-1))\
    * Y_LONGTERM_SIG
y_longterm_sig_correct_2 = np.sqrt(c2_correct/(y_longterm_len-1))\
    * Y_LONGTERM_SIG
print("")
print("Standardabweichung")
print("")
if ((y_longterm_std > y_longterm_sig_warn_1).all(axis=None)
        & (y_longterm_std < y_longterm_sig_warn_2).all(axis=None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
if ((y_longterm_std > y_longterm_sig_correct_1).all(axis=None)
        & (y_longterm_std < y_longterm_sig_correct_2).all(axis=None)):
    print("Eingriffsgrenzen nicht überschritten")
else:
    print("Eingriffsgrenzen überschritten")


# Visualization
fig3 = plt.figure(3, figsize=(12, 4))
fig3.suptitle('')
ax1, ax2 = fig3.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
ax1.plot(np.arange(0, y_longterm.shape[0])+1, y_longterm_mean,
         'bo-', label='Mittelwert')
ax1.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_mean_correct_1*np.ones(y_longterm.shape[0]),
         'r:', label='EG')
ax1.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_mean_correct_2*np.ones(y_longterm.shape[0]), 'r:')
ax1.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_mean_warn_1*np.ones(y_longterm.shape[0]),
         'g--', label='WG')
ax1.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_mean_warn_2*np.ones(y_longterm.shape[0]), 'g--')
#ax1.axis([0, 13, 5.997, 6.0045])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Mittelwert $\overline{T}$ / $^\circ$C')
ax1.set_title('Kontrolle des Mittelwerts')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)
ax2.plot(np.arange(0, y_longterm.shape[0])+1, y_longterm_std, 'bo-',
         label='Standardabweichung')
ax2.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_sig_correct_1*np.ones(y_longterm.shape[0]),
         'r:', label='EG')
ax2.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_sig_correct_2*np.ones(y_longterm.shape[0]), 'r:')
ax2.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_sig_warn_1*np.ones(y_longterm.shape[0]), 'g--', label='WG')
ax2.plot(np.arange(0, y_longterm.shape[0])+1,
         y_longterm_sig_warn_2*np.ones(y_longterm.shape[0]), 'g--')
#ax2.axis([0, 13, 0, 0.004])
ax2.set_xlabel('Stichprobe')
ax2.set_ylabel('Standardabweichung s / $^\circ$C')
ax2.set_title('Kontrolle der Standardabweichung')
ax2.grid(True)
ax2.legend(loc=9, ncol=3)
