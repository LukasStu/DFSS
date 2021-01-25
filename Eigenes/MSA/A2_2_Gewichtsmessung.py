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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

""" a) Die Einrichtung besitzt eine Auflösung von 2 Nachkommastellen. Bewerten Sie, welche Grenze
 sich hinsichtlich der Zieltoleranz aus dieser Auflösung ergibt."""
 
Y_RESOLUTION = 0.01
Y_RICHTWERT = 0.05
 
y_tolerance_resolution = Y_RESOLUTION / Y_RICHTWERT
print("2. Systematische Abweichung und Wiederholbarkeit\n")
print("Die Zieltoleranz beträgt {:.3f}".format(y_tolerance_resolution))

"""b) Bewerten Sie die systematische Messabweichung. Stellen Sie dazu die Stichprobe im Vergleich zu
dem Referenzwert grafisch dar. Berechnen Sie aus den erforderlichen Qualitätskenngrößen die Toleranz,
die sich bei einem Cg und Cgk-Wert von 1.33 ergibt. Führen Sie außerdem einen Hypothesentest
auf systematische Messabweichung durch."""

# Systematic Deviation and Repeatability

# Load data set and reference
data = loadmat('GewichtVerfahren1')
y_repeat_test = data["Messwerte"]
y_repeat_len = np.size(y_repeat_test)
Y_REPEAT_REFERENCE = 147.35

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(np.arange(0, y_repeat_len)+1, y_repeat_test, 'bo-',label="Messwerte")
ax1.plot(np.arange(0, y_repeat_len)+1, Y_REPEAT_REFERENCE*np.ones(y_repeat_len), 'r', label="Referenzwert")
#ax1.axis([0, 26, 4.994, 5.006])
ax1.set_xlabel('Messung')
ax1.set_ylabel('Gewicht $m$ / g')
ax1.set_title('Visualisierung der systematischen Messabweichung')
ax1.legend()
ax1.grid(True)

# Calculation of capability index

C_G = 1.33
y_tolerance_repeatability = C_G*3*np.std(y_repeat_test, ddof=1)*10
print("")
print("")
print("1. Auflösung und Toleranz")
print("")
print("C_g = 1.33 begrenzt Toleranz auf:", round(y_tolerance_repeatability, 3))

C_GK = 1.33
y_deviation = np.mean(y_repeat_test) - Y_REPEAT_REFERENCE
y_tolerance_deviation = (C_GK*3*np.std(y_repeat_test, ddof=1)
                         + np.abs(y_deviation))*10
print("")
print("C_gk = 1.33 begrenzt Toleranz auf ", round(y_tolerance_deviation, 3))

# Hypothesistest with H0: y_repeat_test = Y_REPEAT_REFERENCE
# Test auf gleiche Mittelwerte
hypo_test = stats.ttest_1samp(y_repeat_test, Y_REPEAT_REFERENCE)
print("")
print("Hypothesentest auf Abweichung mit p-value = ",
      round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")
    
"""c) Bewerten Sie die Linearität des Messsystems. Stellen Sie dazu die Stichprobe im Vergleich zu dem
Referenzwert grafisch dar. Führen Sie einen Hypothesentest auf Linearität durch. Welche Grenze
hinsichtlich der Zieltoleranz ergibt sich aus der Untersuchung."""

# Datensatz für Linearitätsuntersuchung
data = loadmat('GewichtVerfahren4')
y_linearity = pd.DataFrame({'reference': np.repeat([98.50, 123.17, 147.35, 172.08, 196.82],10),
                            'value': np.reshape(data["Messwerte"], -1, order='F')})
y_linearity['deviation'] = y_linearity['value'] - y_linearity['reference']

# Visualization
fig2 = plt.figure(2, figsize=(12, 4))
fig2.suptitle('')
ax1, ax2 = fig2.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
ax1.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
ax1.axis([90, 200, -2, 2]) #TODO: Change reference range and tolerance range
ax1.set_xlabel('Referenzwert')
ax1.set_xlabel('Referenzwert $m$ / g')
ax1.set_ylabel(r' Abweichung $\Delta m$ / g')
ax1.set_title('Bewertung des Konfidenzbereichs')
ax1.grid(True)
ax2.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
ax2.axis([90, 200, -2, 2]) #TODO Use the same settings as above
ax2.set_xlabel('Referenzwert $m$ / g')
ax2.set_ylabel(r' Abweichung $\Delta m$ / g')
ax2.set_title('Mittelwerte zur Lineartätsbewertung')
ax2.grid(True)

# Regression function with confidence bounds
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
print(model.summary())
y_plot = np.arange(100, 200, 1) # TODO: change range here
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
print("Prüfung Regressionsgerade")
if (model.pvalues > 0.05).all(axis=None):
    print("Keine signifikante Abweichung zur Linearität")
else:
    print("Signifikante Abweichung zur Linearität")
    
# Position of mean values for each reference
ax2.plot(y_linearity.groupby("reference").agg('mean'), 'ro')
print("")
# Grenze hinsichtlich der Zieltoleranz
y_tolerance_linearity = np.max(np.abs(y_linearity.groupby("reference")
                               .agg("mean"))["deviation"]*20)
print("Linearität begrenzt Toleranz auf:", round(y_tolerance_linearity, 3))

"""d) Bewerten Sie das Streuverhalten des Messsystems. Stellen Sie dazu
 die Messergebnisse in einer geeigneten Form dar. Wie beeinflusst ein
 %GRR-Wert von 30 % die Grenze der Zieltoleranz? ->GRR = 0.3 annehmen"""
 
# Assessment of process variation according to prodedure 3

# Load, format and evaluate data
data = loadmat('GewichtVerfahren3.mat')

Y_K = 25 # TODO: Enter number of objects
Y_N = 2  # TODO: Enter number of measurements series

y_variation_3 = pd.DataFrame({'Part': np.tile(np.arange(0, Y_K, 1), 2),
                              'Measurement': np.repeat(np.arange(1,Y_N+1), 25),
                              'Value': data["Daten"].reshape(-1, order='F')})

# Calculation of normalized squares sums making use of anova table
model = ols('Value ~ C(Part)', data=y_variation_3).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
anova1["M"] = anova1["sum_sq"]/anova1["df"]

# estimations of variance and calculation of GRR and ndc
equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
part_variation = np.sqrt((anova1.loc["C(Part)", "M"]
                          - anova1.loc["Residual", "M"])/Y_N)
grr = equipment_variation
y_tolerance_grr = 6*grr/0.3
ndc = 1.41*part_variation/grr
print("")
print("")
print("4. Streuverhalten: Verfahren 3")
print("")
print("Relativer GRR-Wert 0.3 begrenzt Toleranz auf:",
      round(y_tolerance_grr, 3))
print("Number of Distict Categories ndc = ", round(ndc, 3))

# Visualization
y_variation_3_multi\
    = y_variation_3.set_index(['Measurement', 'Part'])
fig6 = plt.figure(6, figsize=(6, 4))
fig6.suptitle('')
ax1 = fig6.subplots(1, 1)
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],
         'b', label='Messung 1')
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],
         'r:', label='Messung 2')
#ax1.axis([0, 26, 5.96, 6.06])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel('Gewicht $m$ / g')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)

"""f) Bewerten Sie die Langzeitstabilität des Messsystems mithilfe der
Shewhart Regelkarte. Zeichnen Sie eine Eingriffsgrenze (gamma = 99 %) und
eine Warngrenze (gamma = 95 %) ein."""

# Long term stability

# Load and evaluate data
data = loadmat('GewichtVerfahren5')
y_longterm = data["Messwerte"].T
Y_LONGTERM_MU = 123.38 # TODO: Enter mu
Y_LONGTERM_SIG = 0.419 # TODO: Enter sigma
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

"""g) Für welche Zieltoleranz ist das Messsystem gerade eben noch fähig?"""

target = np.max([y_tolerance_resolution,
                 y_tolerance_repeatability,
                 y_tolerance_deviation,
                 y_tolerance_linearity,
                 y_tolerance_grr])
print("Das Messsystem ist für eine Zieltoleranz von {:.4f}g fähig".format(target))