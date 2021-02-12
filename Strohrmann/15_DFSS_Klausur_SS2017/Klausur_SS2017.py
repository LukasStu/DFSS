# -*- coding: utf-8 -*-

""" Musterlösung zur Klausur Design For Six Sigma SS2017
Umsetzung in Python im WS 2020/21

Update on sun Jan 10 2021
@author: stma0003
"""

from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sympy as syms
from scipy.stats import norm
from scipy.stats import t

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


""" Aufgabenteil a: Grafische Darstellung der Messwerte """

# Load and format data
data = loadmat('Temperaturabhaengigkeit')
y_regress = pd.DataFrame({'temperature': data["T"].reshape(-1),
                          'gamma': data["gamma"].reshape(-1)})

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(y_regress["temperature"], y_regress["gamma"], 'bo',
         label='Messwerte')
ax1.axis([20, 30, 0.85, 1.15])
ax1.set_xlabel('Temperatur $T$ / °C')
ax1.set_ylabel(r'Leitfähigkeit $\gamma$ ')
ax1.set_title('Visualisierung der temperaturabhängigen Leitfähigkeit')
ax1.grid(True)


""" Aufgabenteil b: Regressionsfunktion für die Messwerte """

GAMMA = 0.99
# Regression function of conductivity als a function of temperature
poly = ols("gamma ~ temperature + I(temperature**2)", y_regress)
model = poly.fit()
print(model.summary())

# Quadratic term is not significant, p-value = 0.16 > 0.05
poly = ols("gamma ~ temperature", y_regress)
model = poly.fit()

# Regression function and confidence bounds, predction interval not required
y_plot = pd.DataFrame({"temperature": np.arange(20, 31, 1)})
y_plot["gamma"] = model.predict(y_plot)
y_plot["confidence"], y_plot["prediction"] = \
    conf_pred_band_ex(y_plot, poly, model, alpha=1-GAMMA)

# Visualization
ax1.plot(y_plot["temperature"], y_plot["gamma"], 'r',
         label='Regressionsfunktion')
ax1.plot(y_plot["temperature"],
         y_plot["gamma"]+y_plot["confidence"], 'r:',
         label='Konfidenzbereich')
ax1.plot(y_plot["temperature"],
         y_plot["gamma"]-y_plot["confidence"], 'r:')
ax1.plot(y_plot["temperature"],
         y_plot["gamma"]+y_plot["prediction"], 'g--',
         label='Prognosebereich')
ax1.plot(y_plot["temperature"],
         y_plot["gamma"]-y_plot["prediction"], 'g--')
ax1.legend(loc=9, ncol=2)


""" Aufgabenteil c: Konfidenzbereich der Regressionskoeffizienten """

# Confidence bound for regression coefficients
print("")
print("")
print("Konfidenzbereich der Regressionskoeffizienten ")
print("alpha = 99 %")
print(model.conf_int(alpha=1-GAMMA))


""" Aufgabenteil d: Herleitung siehe Musterlösung """


""" Aufgabenteil e: Statistische Tolerierung über den Grenzwertsatz """

# Data according to problem
ALPHA = 2.1e-2
TEMP0 = 25
U0 = 10
gamma01_0 = 1
gamma01_min = 1 - 0.025
gamma01_max = 1 + 0.025
gamma01_tolerance = (gamma01_max - gamma01_min)
gamma01_sig = gamma01_tolerance/np.sqrt(12)
temp1_0 = 25
temp1_sig = 0.03
temp2_0 = 25
temp2_sig = 0.03
temp_roh = 0.8
ur1_0 = 0.2
ur1_sig = 0.01e-3
ur2_0 = 0.2
ur2_sig = 0.01e-3
r1_0 = 10e3
r1_sig = 0.05
r2_0 = 10e3
r2_sig = 0.05
r_roh = 0.95

# Definition of symbolic variables and function
# U1 = U2, reduction of fraction
gamma01, gamma02, temp1, temp2, ur1, ur2, r1, r2 = \
    syms.symbols('gamma01, gamma02, temp1, temp2, ur1, ur2, r1, r2')
gamma02 = gamma01*(1+ALPHA*(temp1-TEMP0))/(1+ALPHA*(temp2-TEMP0))*ur2/ur1*r1/r2

# Symbolic calculation of sensitivities
E_gamma01 = gamma02.diff(gamma01)
E_temp1 = gamma02.diff(temp1)
E_temp2 = gamma02.diff(temp2)
E_ur1 = gamma02.diff(ur1)
E_ur2 = gamma02.diff(ur2)
E_r1 = gamma02.diff(r1)
E_r2 = gamma02.diff(r2)

# Substitute symbols by values, numeric calculation of sensitivities
values = {gamma01: gamma01_0, temp1: temp1_0, temp2: temp2_0,
          ur1: ur1_0, ur2: ur2_0, r1: r1_0, r2: r2_0}
Egamma01 = float(E_gamma01.evalf(subs=values))
Etemp1 = float(E_temp1.evalf(subs=values))
Etemp2 = float(E_temp2.evalf(subs=values))
Eur1 = float(E_ur1.evalf(subs=values))
Eur2 = float(E_ur2.evalf(subs=values))
Er1 = float(E_r1.evalf(subs=values))
Er2 = float(E_r2.evalf(subs=values))

# Tolerance calculation according central limit theorem clt
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
gamma02_tolerance_clt = (c2 - c1) *\
    np.sqrt((Egamma01*gamma01_sig)**2
            + (Etemp1*temp1_sig)**2 + (Etemp2*temp2_sig)**2
            + 2*temp_roh*Etemp1*Etemp2*temp1_sig*temp2_sig
            + (Eur1*ur1_sig)**2 + (Eur2*ur2_sig)**2
            + (Er1*r1_sig)**2 + (Er2*r2_sig)**2
            + 2*r_roh*Er1*Er2*r1_sig*r2_sig)
print("")
print("Toleranz bei Grenzwertmethode:", gamma02_tolerance_clt)


""" Aufgabenteil f: Statistische Tolerierung über Monte Carlo Simulation """

# Generation of random numbers according to specified distribution
N = 10000
gamma01_sim = np.random.uniform(gamma01_min, gamma01_max, N)
ur1_sim = np.random.normal(ur1_0, ur1_sig, N)
ur2_sim = np.random.normal(ur2_0, ur2_sig, N)
z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
temp1_sim = temp1_0 + temp1_sig*z1
temp2_sim = temp2_0 + temp_roh*temp2_sig*z1\
    + np.sqrt(1-temp_roh**2)*temp2_sig*z2
z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
r1_sim = r1_0 + r1_sig*z1
r2_sim = r2_0 + r_roh*r2_sig*z1 + np.sqrt(1 - r_roh**2)*r2_sig*z2
gamma02_sim = gamma01_sim*(1+ALPHA*(temp1_sim-TEMP0))\
    / (1+ALPHA*(temp2_sim-TEMP0))*ur2_sim/ur1_sim*r1_sim/r2_sim

# Tolerance calculation according monte carlo simulation mcs
c1 = t.ppf((1-GAMMA)/2, N-1)
c2 = t.ppf((1+GAMMA)/2, N-1)
gamma02_tolerance_mcs = np.std(gamma02_sim)*np.sqrt(1+1/N)*(c2-c1)
print(' ')
print('Toleranzbereich bei Monte-Carlo-Simulation : ', gamma02_tolerance_mcs)


""" Aufgabenteil g: Auswirkung der Korrelation """

# Analysis of correlation related addend
print("")
print("Verringerung der Toleranz durch die  Korrelation der Temperaturen:",
      2*temp_roh*Etemp1*Etemp2*temp1_sig*temp2_sig)
print("Verringerung der Toleranz durch die Korrelation der Widerstände:",
      2*r_roh*Er1*Er2*r1_sig*r2_sig)
print("Aufgrund der Korrelation wird die Toleranz verringert.")


""" Aufgabenteil h: Hypothesentest auf definierte Leitfähigkeit """

# Standarddeviation of poulation is known, z-test to gamma0 = 1
# calculation interval for accpetance of hypothesis
HYP_GAMMA0 = 1
HYP_ALPHA = 0.1
HYP_SIG = 0.005
c1 = norm.ppf(HYP_ALPHA/2)
c2 = norm.ppf(1 - HYP_ALPHA/2)
hyp_limit_1 = HYP_GAMMA0 + c1*HYP_SIG
hyp_limit_2 = HYP_GAMMA0 + c2*HYP_SIG
print("")
print("Untere Annahmegrenze des Hypothesentests:", hyp_limit_1)
print("Obere Annahmegrenze des Hypothesentests:", hyp_limit_2)


""" Aufgabenteil i: Gütefunktion für Hypothesentest """

# Standarddeviation of poulation is known, z-test to gamma0 = 1
# calculation interval for accpetance of hypothesis
hyp_gamma_var = np.arange(0.95, 1.051, 0.001)
hyp_guete = norm.cdf((hyp_limit_1 - hyp_gamma_var)/HYP_SIG)\
    + 1 - norm.cdf((hyp_limit_2 - hyp_gamma_var)/HYP_SIG)
index_limit = np.min(np.where(hyp_guete <= 0.99))
gamma0_delta_limit = HYP_GAMMA0 - hyp_gamma_var[index_limit]
print("")
print("Mit einer Wahrscheinlichkeit von 99 % erkennbare Abweichung:",
      gamma0_delta_limit)

# Visualization for plausibilization
fig2 = plt.figure(2, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig2.subplots(1, 1)
ax1.plot(hyp_gamma_var, hyp_guete, 'b')
ax1.axis([0.95, 1.05, 0, 1])
ax1.set_xlabel(r'Leitfähigkeit $\gamma$ ')
ax1.set_ylabel('Gütefunktion')
ax1.grid(True)


""" Aufgabenteil j: Systematische Abweichung bei MSA  """


# Systematic Deviation and Repeatability
Y_REPEAT_REFERENCE = 1
Y_TOLERANCE = 0.1

# Load data set and reference
data = loadmat('SystematischerMessfehler')
y_repeat_test = data["gammaSys"].reshape(-1)
y_repeat_len = np.size(y_repeat_test)

# Visualization
fig3 = plt.figure(3, figsize=(6, 4))
fig3.suptitle('')
ax1 = fig3.subplots(1, 1)
ax1.plot(np.arange(0, y_repeat_len)+1, y_repeat_test, 'bo-')
ax1.plot(np.arange(0, y_repeat_len)+1,
         Y_REPEAT_REFERENCE*np.ones(y_repeat_len), 'r')
ax1.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE+0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
ax1.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE-0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
ax1.axis([0, 51, 0.97, 1.03])
ax1.set_xlabel('Messung')
ax1.set_ylabel(r'Leitfähigkeit $\gamma$ / $\mu$S')
ax1.set_title('Visualisierung der systematischen Messabweichung')
ax1.grid(True)

# Calculation of capability index
y_deviation = np.mean(y_repeat_test) - Y_REPEAT_REFERENCE
c_g = 0.1*Y_TOLERANCE/3/np.std(y_repeat_test, ddof=1)
print("")
print("")
print("Systematische Abweichung und Wiederholbarkeit")
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

# Confidence bounds für y_repeat_test
GAMMA = 0.95
c1 = stats.t.ppf((1-GAMMA)/2, y_repeat_len-1)
c2 = stats.t.ppf((1+GAMMA)/2, y_repeat_len-1)
y_repeat_min = np.mean(y_repeat_test) + c1*np.std(y_repeat_test, ddof=1)\
    / np.sqrt(y_repeat_len)
y_repeat_max = np.mean(y_repeat_test) + c2*np.std(y_repeat_test, ddof=1)\
    / np.sqrt(y_repeat_len)
print("")
print("Konfidenzbereich: Untere Grenze = ", round(y_repeat_min, 4))
print("Konfidenzbereich: Obere Grenze = ", round(y_repeat_max, 4))
if (Y_REPEAT_REFERENCE >= y_repeat_min) & (Y_REPEAT_REFERENCE <= y_repeat_max):
    print("Abweichung nicht signifikant")
else:
    print("Abweichung signifikant")


""" Aufgabenteil k: Streuverhalten nach Verfahren 3  """

# Load, format and evaluate data
data = loadmat('Reproduzierbarkeit')
y_variation_3 = pd.DataFrame({'Part': np.tile(np.arange(0, 25, 1), 2),
                              'Measurement': np.repeat([1, 2], 25),
                              'Value': np.reshape([data["gammaRep1"],
                                                   data["gammaRep2"]], -1)})
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
print("Streuverhalten: Verfahren 3 ohne Temperaturkompensation")
print("")
print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
print("Number of Distict Categories ndc = ", round(ndc, 3))

# Visualization
y_variation_3_multi\
    = y_variation_3.set_index(['Measurement', 'Part'])
fig4 = plt.figure(4, figsize=(12, 4))
fig4.suptitle('')
ax1, ax2 = fig4.subplots(1, 2)
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],
         'b', label='Messung 1')
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],
         'r:', label='Messung 2')
ax1.axis([0, 26, 0.9, 1.1])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Leitfähigkeit $\gamma$ / $\mu$S')
ax1.set_title('Streuverhalten ohne Temperaturkompensation')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)


""" Aufgabenteil l: Streuverhalten mit Temperaturkompensation """

# Load, format and evaluate data
y_variation_3 = pd.DataFrame({'Part': np.tile(np.arange(0, 25, 1), 2),
                              'Measurement': np.repeat([1, 2], 25),
                              'Value': np.reshape([data["gammaRep1"],
                                                   data["gammaRep2"]], -1)})
temp = np.reshape([data["T1"], data["T2"]], -1)
y_variation_3["Value"] = y_variation_3["Value"] / (1 + ALPHA*(temp - TEMP0))
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
print("Streuverhalten: Verfahren 3 mit Temperaturkompensation")
print("")
print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
print("Number of Distict Categories ndc = ", round(ndc, 3))

# Visualization
y_variation_3_multi\
    = y_variation_3.set_index(['Measurement', 'Part'])
ax2.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],
         'b', label='Messung 1')
ax2.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],
         'r:', label='Messung 2')
ax2.axis([0, 26, 0.9, 1.1])
ax2.set_xlabel('Stichprobe')
ax2.set_ylabel(r'Leitfähigkeit $\gamma$ / $\mu$S')
ax2.set_title('Streuverhalten mit Temperaturkompensation')
ax2.grid(True)
ax2.legend(loc=9, ncol=3)


""" Aufgabenteil m: Langezeitstabilität  """

# Load and evaluate data
data = loadmat('Langzeitstabilitaet')
y_longterm = data["gammaLang"].T
Y_LONGTERM_MU = Y_REPEAT_REFERENCE
Y_LONGTERM_SIG = HYP_SIG
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
print("Langzeitstabilität")
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
fig5 = plt.figure(5, figsize=(12, 4))
fig5.suptitle('')
ax1, ax2 = fig5.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
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
ax1.axis([0, 11, 0.98, 1.02])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Mittelwert $\overline{\gamma}$ / $\mu$S')
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
ax2.axis([0, 11, 0, 0.01])
ax2.set_xlabel('Stichprobe')
ax2.set_ylabel(r'Standardabweichung s / $\mu$S')
ax2.set_title('Kontrolle der Standardabweichung')
ax2.grid(True)
ax2.legend(loc=9, ncol=3)


