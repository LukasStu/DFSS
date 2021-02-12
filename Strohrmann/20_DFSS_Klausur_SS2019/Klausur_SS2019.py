# -*- coding: utf-8 -*-

""" Musterlösung zur Klausur Design For Six Sigma SS2019
Umsetzung in Python im WS 2020/21

Update on sun Jan 24 2021
@author: stma0003
"""

from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import t
import sympy as syms
from scipy.stats import norm

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


""" Aufgabenteil a: Hypothesentest auf gleichen Mittelwert """

# Load and format data
data = loadmat('Signifikanz')
m_hyp = pd.DataFrame({'m1': data["M1"].reshape(-1),
                      'm2': data["M2"].reshape(-1),
                      'n1': data["n1"].reshape(-1),
                      'n2': data["n2"].reshape(-1)})

# t-test due to unknown variance, difference of means
GAMMA = 0.95
N = m_hyp.m1.shape[0]
c1 = t.ppf((1 - GAMMA)/2, 2*N - 2)
c2 = t.ppf((1 + GAMMA)/2, 2*N - 2)
m_hyp_diff = np.mean(m_hyp.m1 - m_hyp.m2)
m_hyp_s1 = np.std(m_hyp.m1, ddof=1)
m_hyp_s2 = np.std(m_hyp.m2, ddof=1)
m_hyp_s = np.sqrt((m_hyp_s1**2 + m_hyp_s2**2) / 2)
m_hyp_c1 = c1*np.sqrt(2/N)*m_hyp_s
m_hyp_c2 = c2*np.sqrt(2/N)*m_hyp_s
print("")
print("Abweichung der Stichprobe =", round(m_hyp_diff, 4))
print("Unterer Annahmegrenze =", round(m_hyp_c1, 4))
print("Obere Annahmegrenze =", round(m_hyp_c2, 4))
if ((m_hyp_c1 <= m_hyp_diff) & (m_hyp_diff <= m_hyp_c2)):
    print("Abweichung nicht signifikant")
else:
    print("Abweichung signifikant")


""" Aufgabenteil b: Gütefunktion"""

# Definition of variable difference for population
m_hyp_mudiff = np.arange(-1, 1.01, 0.01)
m_hyp_guete = t.cdf((m_hyp_c1 - m_hyp_mudiff)/np.sqrt(2/N)/m_hyp_s, 2*N-2)\
    + 1 - t.cdf((m_hyp_c2 - m_hyp_mudiff)/np.sqrt(2/N)/m_hyp_s, 2*N-2)

# Visualization of quality function
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
ax1.plot(m_hyp_mudiff, m_hyp_guete, 'b')
ax1.axis([-1, 1, 0, 1])
ax1.set_xlabel(r'Abweichung ${\Delta}M$ / N${\cdot}$m')
ax1.set_ylabel('Gütefunktion')
ax1.grid(True)

# Determination of 99 % limit
index_min = np.min(np.where(m_hyp_guete <= 0.99))
print("Abweichung, die mit P = 99 % erkannt wird =",
      round(m_hyp_mudiff[index_min], 4))


""" Aufgabenteil c: Grafische Plausibilisierung """

# Visualization
fig2 = plt.figure(2, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig2.subplots(1, 1)
boxplot = m_hyp.boxplot(column=['m1', 'm2'], ax=ax1)
ax1.axis([0, 3, -13.25, -12.75])
ax1.set_xlabel('Gruppe')
ax1.set_ylabel(r'Schleppmoment $M$ / N$\cdot$m')
ax1.grid(True)
print("")
print("Boxen des Box-Plot überlappen, Grafik bestätigt, dass keine")
print("signifikante Abweichung vorliegt.")


""" Aufgabenteil d: Tolerierung auf Basis Grenzwertsatz """

# Correlation test H0: roh = 0, H1: roh <> 0
m_corr = pd.DataFrame({'m': np.append(data["M1"].reshape(-1),
                                      data["M2"].reshape(-1)),
                       'n': np.append(data["n1"].reshape(-1),
                                      data["n2"].reshape(-1))})
Corr, p = stats.pearsonr(m_corr['m'], m_corr['n'])
print()
print('Korrelation zwischen den Größen: ', Corr)
print('p-value zur Korrelation : ', round(p, 5))
if (p <= 0.05):
    print("Korrelation signifikant")
else:
    print("Korrelation nicht signifikant")





""" Aufgabenteil e: Test auf Korrelation """

# Data according to problem
GAMMA = 0.9973
R = 287.1
p_0 = 101300
p_sig = 0.01*p_0/6
t_0 = 323
t_sig = 2/6
eta_0 = 0.8
eta_min = 0.78
eta_max = 0.82
eta_tol = eta_max - eta_min
eta_sig = eta_tol/np.sqrt(12)
vh_0 = 1.9e-3
vh_min = vh_0 - 5e-6
vh_max = vh_0 + 5e-6
vh_tol = vh_max - vh_min
vh_sig = vh_tol/np.sqrt(12)
n_0 = 2500*60
n_min = n_0*0.997
n_max = n_0*1.003
n_tol = n_max - n_min
n_sig = n_tol / np.sqrt(12)

# Definition of symbolic variables and function
dmdt_sym, p_sym, t_sym, eta_sym, vh_sym, n_sym = \
    syms.symbols('dmdr_sym, p_sym, t_sym, eta_sym, vh_sym, n_sym')
dmdt_sym = p_sym/R/t_sym*eta_sym*vh_sym*n_sym

# Symbolic calculation of sensitivities
e_p_sym = dmdt_sym.diff(p_sym)
e_t_sym = dmdt_sym.diff(t_sym)
e_eta_sym = dmdt_sym.diff(eta_sym)
e_vh_sym = dmdt_sym.diff(vh_sym)
e_n_sym = dmdt_sym.diff(n_sym)

# Substitute symbols by values, numeric calculation of sensitivities
values = {p_sym: p_0, t_sym: t_0, eta_sym: eta_0,
          vh_sym: vh_0, n_sym: n_0}
dmdt_0 = float(dmdt_sym.evalf(subs=values))
e_p = float(e_p_sym.evalf(subs=values))
e_t = float(e_t_sym.evalf(subs=values))
e_eta = float(e_eta_sym.evalf(subs=values))
e_vh = float(e_vh_sym.evalf(subs=values))
e_n = float(e_n_sym.evalf(subs=values))

# Tolerance calculation according central limit theorem clt
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
dmdt_tolerance_clt = (c2 - c1) *\
    np.sqrt((e_p*p_sig)**2 + (e_t*t_sig)**2 + (e_eta*eta_sig)**2
            + (e_vh*vh_sig)**2 + (e_n*n_sig)**2)
print("")
print("Luftmassemstrom im Arbeitspunkt = ", round(dmdt_0, 3))
print("Toleranz bei Grenzwertmethode =", round(dmdt_tolerance_clt, 3))


""" Aufgabenteil l: Statistische Toleranzsimulation """

# Generation of random numbers according to specified distribution
N = 10000
p_sim = np.random.normal(p_0, p_sig, N)
t_sim = np.random.normal(t_0, t_sig, N)
eta_sim = np.random.uniform(eta_min, eta_max, N)
vh_sim = np.random.uniform(vh_min, vh_max, N)
n_sim = np.random.uniform(n_min, n_max, N)
dmdt_sim = p_sim/R/t_sim*eta_sim*vh_sim*n_sim

# Tolerance calculation according monte carlo simulation mcs
c1 = t.ppf((1-GAMMA)/2, N-1)
c2 = t.ppf((1+GAMMA)/2, N-1)
dmdt_tolerance_mcs = np.std(dmdt_sim, ddof=1)*np.sqrt(1+1/N)*(c2-c1)
print('Toleranz bei statistischer Simulation =', round(dmdt_tolerance_mcs, 3))


""" Aufgabenteil m: Vergleich der Ergebnisse und Disussion """

# Comparison to numerical evaluation of simulated data
dmdt_sort = np.sort(dmdt_sim)
dmdt_cdf = np.arange(1, N+1, 1)/N
index_min = np.min(np.where(dmdt_cdf >= (1-GAMMA)/2))
index_max = np.min(np.where(dmdt_cdf >= (1+GAMMA)/2))
dmdt_tol_num = dmdt_sort[index_max] - dmdt_sort[index_min]
print("Toleranz bei numerischer Simulation =", round(dmdt_tol_num, 3))

# Visualization of cdf functions
fig3 = plt.figure(3, figsize=(6, 4))
fig3.suptitle('')
ax1 = fig3.subplots(1, 1)
ax1.plot(dmdt_sort, dmdt_cdf, 'b', label="Numerische Auswertung")
ax1.plot(dmdt_sort, norm.cdf(dmdt_sort, dmdt_0, np.std(dmdt_sim, ddof=1)),
         'r', label="Grenzwertmethode")
ax1.axis([240, 260, 0, 1])
ax1.set_xlabel('Luftmassenstrom $dm/dt$ / kg/h')
ax1.set_ylabel('Wahrscheinlichkeitsverteilung $F(dm/dt)$')
ax1.grid(True)
ax1.legend()
print("Annahme des normalverteilten Luftmassenstroms nicht gerechtfertigt.")


""" Aufgabenteil n: Beiträge zur Gesamttoleranz """

# Tolerance causes and their impact
tol_causes = pd.DataFrame({'portion': np.abs([(e_p*p_sig), (e_t*t_sig),
                                              (e_eta*eta_sig), (e_vh*vh_sig),
                                              (e_n*n_sig)])},
                          index=['p', 'T', 'eta', 'Vh', 'n'])

# Visualization
fig4 = plt.figure(4, figsize=(6, 4))
fig4.suptitle('')
ax1 = fig4.subplots(1, 1)
tol_causes.plot.bar(ax=ax1, color='b', legend=None)
ax1.axis([-1, 5, -1, 4])
ax1.set_xlabel('Toleranzursache')
ax1.set_ylabel('Standardabweichung')
ax1.grid(True)
