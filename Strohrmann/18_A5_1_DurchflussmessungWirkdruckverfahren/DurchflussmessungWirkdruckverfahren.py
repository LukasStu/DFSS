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


""" Aufgabenteil a: Herleitung der Maßkette, Darstellung der Sollkennlinie """

# Data according to problem
K = 0.2
M = 0.04
U0 = 10
Q0 = 0.5
uadc_0 = 5
uadc_tol = 0.5
uadc_sig = uadc_tol/6
uoff_0 = 0.01
uoff_min = 0
uoff_max = 0.02
uoff_tol = 0.02
uoff_sig = uoff_tol/np.sqrt(12)
na_0 = 0
na_min = -1/2048
na_max = 1/2048
na_tol = 2/2048
na_sig = na_tol/np.sqrt(12)

# generate plot data
q_plot = np.arange(0, 1.01, 0.01)
n_plot = M/K**2*q_plot**2*(1+uoff_0/uadc_0) + na_0

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(q_plot, n_plot, 'b')
ax1.axis([0, 1, 0, 1])
ax1.set_xlabel('Durchfluss $Q$ / m³/h')
ax1.set_ylabel('Normierter ADC-Wert $N$')
ax1.grid(True)


""" Aufgabenteil b: Berechnung der Empfindlichkeiten im Arbeitspunkt """

# Definition of symbolic variables and function
uadc_sym, uoff_sym, na_sym, n_sym, q_sym =\
    syms.symbols('uadc_sym, uoff_sym, na_sym, n_sym, q_sym')
n_sym = M/K**2*q_sym**2*(1+uoff_sym/uadc_sym) + na_sym

# Symbolic calculation of sensitivities
e_uadc_sym = n_sym.diff(uadc_sym)
e_uoff_sym = n_sym.diff(uoff_sym)
e_na_sym = n_sym.diff(na_sym)
e_q_sym = n_sym.diff(q_sym)

# Substitute symbols by values, numeric calculation of sensitivities
values = {uadc_sym: uadc_0, uoff_sym: uoff_0, na_sym: na_0,
          q_sym: Q0}
e_uadc = float(e_uadc_sym.evalf(subs=values))
e_uoff = float(e_uoff_sym.evalf(subs=values))
e_na = float(e_na_sym.evalf(subs=values))
e_q = float(e_q_sym.evalf(subs=values))

print()
print("Analytische Berechnung der Empfindlichleiten")
print()
print("Empfindlichkeit UADC =", round(e_uadc, 4))
print("Empfindlichkeit UOFF =", round(e_uoff, 4))
print("Empfindlichkeit NA =", round(e_na, 4))


""" Aufgabenteil c: Simulation der Empfindlichkeiten im Arbeitspunkt """

# Definition of samples according to specified distribution
N = 10000
uadc_sim = np.random.normal(uadc_0, uadc_sig, N)
uoff_sim = np.random.uniform(uoff_min, uoff_max, N)
na_sim = np.random.uniform(na_min, na_max, N)
n_sim = M/K**2*Q0**2*(1+uoff_sim/uadc_sim) + na_sim

# Regression function of conductivity as a function of temperature
y_regress = pd.DataFrame({'uadc': uadc_sim.reshape(-1),
                          'uoff': uoff_sim.reshape(-1),
                          'na': na_sim.reshape(-1),
                          'n': n_sim.reshape(-1)})
poly = ols("n ~ uadc + uoff + na", y_regress)
model = poly.fit()
print()
print("Numerische Berechnung der Empfindlichleiten")
print()
print("Empfindlichkeit UADC =", round(model.params.uadc, 4))
print("Empfindlichkeit UOFF =", round(model.params.uoff, 4))
print("Empfindlichkeit NA =", round(model.params.na, 4))
print()
print("Weitgehende Übereinstimmung der Empfindlichkeiten")


""" Aufgabenteil d: Arithmetische Tolerierung für linearisierte Maßkette """

# Worst case tolerance calculation
n_tol_ari = np.abs(e_uadc*uadc_tol) + np.abs(e_uoff*uoff_tol)\
    + np.abs(e_na*na_tol)
print()
print("Toleranz bei arithmetischer Tolerierung =", round(n_tol_ari/e_q, 4))


""" Aufgabenteil e: Statistische Tolerierung für linearisierte Maßkette """

# Tolerance calculation according central limit theorem clt
GAMMA = 0.9973
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
n_tol_clt_sig = np.sqrt((e_uadc*uadc_sig)**2
                        + (e_uoff*uoff_sig)**2
                        + (e_na*na_sig)**2)
n_tol_clt = (c2 - c1) * n_tol_clt_sig
print("Toleranz bei Grenzwertmethode =", round(n_tol_clt/e_q, 4))


""" Aufgabenteil f: Statistische Simulation """

# Tolerance calculation based on simulation and normal distrituted result
n_mean = np.mean(n_sim)
n_std = np.std(n_sim, ddof=1)
n_tol_sim = (c2 - c1) * n_std*np.sqrt(1+1/N)
print("Toleranz bei statistischer Simulation =", round(n_tol_sim/e_q, 4))

# Tolerance calculation based on numeric simulation result
n_sort = np.sort(n_sim)
n_cdf = np.arange(1,N+1,1)/N
index_min = np.min(np.where(n_cdf >= (1-GAMMA)/2))
index_max = np.min(np.where(n_cdf >= (1+GAMMA)/2))
n_tol_num = n_sort[index_max] - n_sort[index_min]
print("Toleranz bei numerischer Simulation =", round(n_tol_num/e_q, 4))

# Interpretation of results
print()
print("Durch die Umrechnung der Rechteckverteilung bei Offset-Spannung und ")
print("Auflösung auf eine äquivalente Standardabweichung wird der ")
print("Toleranzbereich mit gamma = 99.73 % vergrößert. Da nur wenige Maße")
print("überlagert werden, ist der Gewinn der statistischen Tolerierung ")
print("gering. Damit ist die Toleranz bei statistischer Tolerierung größer ")
print("als bei arithmetischer Tolerierung. Außerdem ist die Näherung, dass")
print("die Aufangsgröße eine Normalverteilung aufweist, wegen der geringen")
print("Anzahl von Eingangsgrößen nicht erfüllt.")


""" Zusatzaufgabe: Berechnung der Wahrscheinlichkeitsdichte über Faltung """

# lineare Masskette: n_tol = e_uadc*uadc_tol + e_uoff*uoff_tol + e_na*na_tol
N_RES = 1e-7
n_uadc = np.arange(-0.002, 0.002+N_RES, N_RES)
pdf_uadc = norm.pdf(n_uadc, 0, np.abs(e_uadc*uadc_sig))
n_uoff = np.arange(-0.002, 0.002+N_RES, N_RES)
pdf_uoff = stats.uniform.pdf(n_uadc, e_uoff*uoff_min, e_uoff*uoff_tol)
n_uadc_uoff = np.arange(-0.004, 0.004+N_RES, N_RES)
pdf_uadc_uoff = np.convolve(pdf_uadc, pdf_uoff)*N_RES
n_na = np.arange(-0.002, 0.002+N_RES, N_RES)
pdf_na = stats.uniform.pdf(n_na, e_na*na_min, e_na*na_tol)
n_uadc_uoff_na = np.arange(-0.006, 0.006+N_RES, N_RES)
pdf_uadc_uoff_na = np.convolve(pdf_uadc_uoff, pdf_na)*N_RES
cdf_uadc_uoff_na = np.cumsum(pdf_uadc_uoff_na)*N_RES
cdf_uadc_uoff_na = cdf_uadc_uoff_na / np.max(cdf_uadc_uoff_na)
index_min = np.min(np.where(cdf_uadc_uoff_na >= (1-GAMMA)/2))
index_max = np.min(np.where(cdf_uadc_uoff_na >= (1+GAMMA)/2))
n_tol_con = n_uadc_uoff_na[index_max] - n_uadc_uoff_na[index_min]

print ()
print("Toleranz bei Faltung =", round(n_tol_con/e_q, 4))


# Visualization
fig2 = plt.figure(2, figsize=(6, 4))
fig2.suptitle('')
ax1 = fig2.subplots(1, 1)
ax1.plot(n_uadc_uoff_na, pdf_uadc_uoff_na, 'b', label="Faltung")
ax1.plot(n_uadc_uoff_na+0.0005,
         norm.pdf(n_uadc_uoff_na, 0, n_tol_clt_sig),
         'r', label="Grenzwertmethode")
ax1.axis([-0.001, 0.002, 0, 1200])
ax1.set_xlabel('ADC-Einheiten $n$')
ax1.set_ylabel('Wahrscheinlichkeitsdichte $f(n)$')
ax1.grid(True)
ax1.legend()
