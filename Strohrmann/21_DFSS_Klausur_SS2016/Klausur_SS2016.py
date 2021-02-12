# -*- coding: utf-8 -*-

""" Musterlösung zur Klausur Design For Six Sigma SS2016
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
from scipy.stats import chi2


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


""" Aufgabenteil a: Schätzung von Parametern """

# Load and format data
data = loadmat('Widerstandsmessung')
resistor = pd.DataFrame({'ru': data["RU"].reshape(-1),
                         'rr': data["RR"].reshape(-1)})

# sample parameters
ru_mean = np.mean(resistor.ru)
ru_sig = np.std(resistor.ru, ddof=1)
rr_mean = np.mean(resistor.rr)
rr_sig = np.std(resistor.rr, ddof=1)

# confidence bounds for mean, t variable due to unknown variance
GAMMA = 0.95
n = resistor.ru.shape[0]
c1 = t.ppf((1 - GAMMA)/2, n - 1)
c2 = t.ppf((1 + GAMMA)/2, n - 1)
ru_mu_min = ru_mean + c1*ru_sig/np.sqrt(n)
ru_mu_max = ru_mean + c2*ru_sig/np.sqrt(n)
rr_mu_min = rr_mean + c1*rr_sig/np.sqrt(n)
rr_mu_max = rr_mean + c2*rr_sig/np.sqrt(n)
print("")
print("Mittelwerte mit Konfidenzbereich")
print("Widerstand R_U:", round(ru_mu_min, 4), '<=',
      round(ru_mean, 4), '<=', round(ru_mu_max, 4))
print("Widerstand R_R:", round(rr_mu_min, 4), '<=',
      round(rr_mean, 4), '<=', round(rr_mu_max, 4))

# confidence bounds for standard deviation
c1 = chi2.ppf((1 - GAMMA)/2, n - 1)
c2 = chi2.ppf((1 + GAMMA)/2, n - 1)
ru_sig_min = ru_sig*np.sqrt((n - 1)/c2)
ru_sig_max = ru_sig*np.sqrt((n - 1)/c1)
rr_sig_min = rr_sig*np.sqrt((n - 1)/c2)
rr_sig_max = rr_sig*np.sqrt((n - 1)/c1)
print("")
print("Standardabweichungen mit Konfidenzbereich")
print("Widerstand R_U:", round(ru_sig_min, 4), '<=',
      round(ru_sig, 4), '<=', round(ru_sig_max, 4))
print("Widerstand R_R:", round(rr_sig_min, 4), '<=',
      round(rr_sig, 4), '<=', round(rr_sig_max, 4))


""" Aufgabenteil b: Histogramm und Wahrscheinlichkeitsdichte """

ru_plot = np.arange(3.5, 6.51, 0.01)
rr_plot = np.arange(7.5, 12.51, 0.01)
fig1 = plt.figure(1, figsize=(12, 4))
fig1.suptitle('')
ax1, ax2 = fig1.subplots(1, 2)
ax1.hist(resistor.ru, density=True, facecolor='b')
ax1.plot(ru_plot, norm.pdf(ru_plot, ru_mean, ru_sig), 'r')
ax1.axis([3.5, 6.5, 0, 1])
ax1.set_xlabel(r'Widerstand $R_{U}$ / $\Omega$')
ax1.set_ylabel(r'Relative Häufigkeit $h(R_{U})$ 1/$\Omega$')
ax1.grid(True)
ax2.hist(resistor.rr, 20, density=True, facecolor='b', label='Stichprobe')
ax2.plot(rr_plot, norm.pdf(rr_plot, rr_mean, rr_sig), 'r', label='Verteilung')
ax2.axis([7.5, 12.5, 0, 0.7])
ax2.set_xlabel(r'Widerstand $R_{R}$ / $\Omega$')
ax2.set_ylabel(r'Relative Häufigkeit $h(R_{R})$ / 1/$\Omega$')
ax2.grid(True)
ax2.legend(loc=9, ncol=2)


""" Aufgabenteil c: Korrelation der Widerstandswerte """

# hypothesis test of correlation
corr, p = stats.pearsonr(resistor.ru, resistor.rr)

# confidence bounds for correlation
GAMMA = 0.95
n = np.shape(resistor.ru)[0]
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
corr_min = np.tanh(np.arctanh(corr) - c2/np.sqrt(n-3))
corr_max = np.tanh(np.arctanh(corr) - c1/np.sqrt(n-3))
print("")
print("Korrelation der Widerstandswerte:", round(corr, 4))
print("P-Value der Korrelation:", round(p, 4))
print("Konfidenzbereich:", round(corr_min, 4), '<=',
      round(corr, 4), '<=', round(corr_max, 4))

# visualization of correlation
fig2 = plt.figure(2, figsize=(6, 4))
fig2.suptitle('')
ax1 = fig2.subplots(1, 1)
ax1.plot(resistor.ru, resistor.rr, 'b+')
ax1.axis([3.5, 6.5, 7.5, 12.5])
ax1.set_xlabel(r'Widerstand $R_{U}$ / $\Omega$')
ax1.set_ylabel(r'Widerstand $R_{R}$ / $\Omega$')
ax1.grid(True)


""" Aufgabenteil d: Berechnung der geregelten Temperatur """

# calculation of controlled temperature
IU_0 = 50e-6
IR_0 = 20e-6
R_ALPHA_0 = 3000e-6
RU_0 = 5e3
RR_0 = 10e3
T_0 = 20
TU_0 = 20
tr = (IU_0*RU_0*(1+R_ALPHA_0*(TU_0 - T_0)) - IR_0*RR_0*(1-R_ALPHA_0*T_0))\
    / IR_0 / RR_0 / R_ALPHA_0
print("")
print("Temperatur T_R:", round(tr, 4))


""" Aufgabenteil e: Regression der Temperaturmesswerte """

# import and format data
data = loadmat('Temperaturmessung')
measurement = pd.DataFrame({'tr': data["TR"].reshape(-1),
                            'ki': data["kI"].reshape(-1)})

# Regression function of temperature as a function of current ratio
poly = ols("tr ~ ki + I(ki**2) + I(ki**3)", measurement)
model = poly.fit()
print(model.summary())

# Regression function and predction interval, confidence bounds not required
measurement_plot = pd.DataFrame({"ki": np.arange(2, 4.1, 0.1)})
measurement_plot["tr"] = model.predict(measurement_plot)
measurement_plot["confidence"], measurement_plot["prediction"] = \
    conf_pred_band_ex(measurement_plot, poly, model, alpha=1-GAMMA)

# Visualization
fig3 = plt.figure(3, figsize=(12, 4))
fig3.suptitle('')
ax1, ax2 = fig3.subplots(1, 2)
ax1.plot(measurement["ki"], measurement["tr"], 'bo',
         label='Messwerte')
ax1.axis([2, 4, -50, 550])
ax1.set_xlabel(r'Stromverhältnis $k_I$ ')
ax1.set_ylabel('Temperatur $T$ / °C')
ax1.set_title('Regressionfunktion der Ordnung M = 3')
ax1.grid(True)
ax1.plot(measurement_plot["ki"], measurement_plot["tr"], 'r',
         label='Regressionsfunktion')
ax1.plot(measurement_plot["ki"],
         measurement_plot["tr"]+measurement_plot["confidence"], 'r:',
         label='Konfidenzbereich')
ax1.plot(measurement_plot["ki"],
         measurement_plot["tr"]-measurement_plot["confidence"], 'r:')
ax1.plot(measurement_plot["ki"],
         measurement_plot["tr"]+measurement_plot["prediction"], 'g--',
         label='Prognosebereich')
ax1.plot(measurement_plot["ki"],
         measurement_plot["tr"]-measurement_plot["prediction"], 'g--')
# ax1.legend(loc=2, ncol=1)


""" Aufgabenteil f: Entfernen der nicht signifikanten Terme """

# delete non significant term of highest order
poly = ols("tr ~ ki + I(ki**2)", measurement)
model = poly.fit()
print(model.summary())

# delete non significant term of highest order
poly = ols("tr ~ ki", measurement)
model = poly.fit()
print(model.summary())

# Regression function and predction interval, confidence bounds not required
measurement_plot = pd.DataFrame({"ki": np.arange(2, 4.1, 0.1)})
measurement_plot["tr"] = model.predict(measurement_plot)
measurement_plot["confidence"], measurement_plot["prediction"] = \
    conf_pred_band_ex(measurement_plot, poly, model, alpha=1-GAMMA)

# Visualization
ax2.plot(measurement["ki"], measurement["tr"], 'bo',
         label='Messwerte')
ax2.axis([2, 4, -50, 550])
ax2.set_xlabel(r'Stromverhältnis $k_I$ ')
ax2.set_ylabel('Temperatur $T$ / °C')
ax2.set_title('Regressionfunktion der Ordnung M = 1')
ax2.grid(True)
ax2.plot(measurement_plot["ki"], measurement_plot["tr"], 'r',
         label='Regressionsfunktion')
ax2.plot(measurement_plot["ki"],
         measurement_plot["tr"]+measurement_plot["confidence"], 'r:',
         label='Konfidenzbereich')
ax2.plot(measurement_plot["ki"],
         measurement_plot["tr"]-measurement_plot["confidence"], 'r:')
ax2.plot(measurement_plot["ki"],
         measurement_plot["tr"]+measurement_plot["prediction"], 'g--',
         label='Prognosebereich')
ax2.plot(measurement_plot["ki"],
         measurement_plot["tr"]-measurement_plot["prediction"], 'g--')
ax2.legend(loc=2, ncol=1)


""" Aufgabenteil g: Statisische Tolerierung """

# data according problem
GAMMA = 0.9973
T_0 = 20
TU_0 = 40
IU_0 = 50e-6
IU_TOL = 0.5e-6
iu_min = IU_0 - IU_TOL/2
iu_max = IU_0 + IU_TOL/2
iu_sig = IU_TOL/np.sqrt(12)
IR_0 = 20e-6
IR_TOL = 0.2e-6
ir_min = IR_0 - IR_TOL/2
ir_max = IR_0 + IR_TOL/2
ir_sig = IR_TOL/np.sqrt(12)
RU_0 = 5e3
RU_TOL = 0.5e3
ru_sig = RU_TOL/6
RR_0 = 10e3
RR_TOL = 1e3
rr_sig = RR_TOL/6
R_CORR = 0.9
R_ALPHA_0 = 3000e-6
R_ALPHA_TOL = 300e-6
r_alpha_sig = R_ALPHA_TOL/6

# Definition of symbolic variables and function
tr_sym, iu_sym, ir_sym, rr_sym, ru_sym, r_alpha_sym = \
    syms.symbols('tr_sym, iu_sym, ir_sym, rr_sym, ru_sym, r_alpha_sym')
tr_sym = (iu_sym*ru_sym*(1+r_alpha_sym*(TU_0 - T_0))
          - ir_sym*rr_sym*(1-r_alpha_sym*T_0))\
    / ir_sym / rr_sym / r_alpha_sym

# Symbolic calculation of sensitivities
e_iu_sym = tr_sym.diff(iu_sym)
e_ir_sym = tr_sym.diff(ir_sym)
e_ru_sym = tr_sym.diff(ru_sym)
e_rr_sym = tr_sym.diff(rr_sym)
e_r_alpha_sym = tr_sym.diff(r_alpha_sym)

# Substitute symbols by values, numeric calculation of sensitivities
values = {iu_sym: IU_0, ir_sym: IR_0, ru_sym: RU_0,
          rr_sym: RR_0, r_alpha_sym: R_ALPHA_0}
tr_0 = float(tr_sym.evalf(subs=values))
e_iu = float(e_iu_sym.evalf(subs=values))
e_ir = float(e_ir_sym.evalf(subs=values))
e_ru = float(e_ru_sym.evalf(subs=values))
e_rr = float(e_rr_sym.evalf(subs=values))
e_r_alpha = float(e_r_alpha_sym.evalf(subs=values))

# Tolerance calculation according central limit theorem clt
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
tr_tolerance_clt = (c2 - c1) *\
    np.sqrt((e_iu*iu_sig)**2 + (e_ir*ir_sig)**2 + (e_ru*ru_sig)**2
            + (e_rr*rr_sig)**2 + (e_r_alpha*r_alpha_sig)**2
            + 2*R_CORR*e_ru*ru_sig*e_rr*rr_sig)
print("")
print("Temperatur im Arbeitspunkt = ", round(tr_0, 3))
print("Toleranz bei Grenzwertmethode =", round(tr_tolerance_clt, 3))


""" Aufgabenteil h: Statistische Toleranzsimulation """

# Generation of random numbers according to specified distribution
N = 10000
iu_sim = np.random.uniform(iu_min, iu_max, N)
ir_sim = np.random.uniform(ir_min, ir_max, N)
z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
ru_sim = RU_0 + ru_sig*z1
rr_sim = RR_0 + R_CORR*rr_sig*z1 + np.sqrt(1-R_CORR**2)*rr_sig*z2
r_alpha_sim = np.random.normal(R_ALPHA_0, r_alpha_sig, N)
tr_sim = (iu_sim*ru_sim*(1+r_alpha_sim*(TU_0 - T_0))
          - ir_sim*rr_sim*(1-r_alpha_sim*T_0))\
    / ir_sim / rr_sim / r_alpha_sim

# Tolerance calculation according monte carlo simulation mcs
c1 = t.ppf((1-GAMMA)/2, N-1)
c2 = t.ppf((1+GAMMA)/2, N-1)
tr_tolerance_mcs = np.std(tr_sim, ddof=1)*np.sqrt(1+1/N)*(c2-c1)
print('Toleranz bei statistischer Simulation =', round(tr_tolerance_mcs, 3))

# Comparison to numerical evaluation of simulated data
tr_sort = np.sort(tr_sim)
tr_cdf = np.arange(1, N+1, 1)/N
index_min = np.min(np.where(tr_cdf >= (1-GAMMA)/2))
index_max = np.min(np.where(tr_cdf >= (1+GAMMA)/2))
tr_tol_num = tr_sort[index_max] - tr_sort[index_min]
print("Toleranz bei numerischer Simulation =", round(tr_tol_num, 3))

# Visualization of cdf functions
fig4 = plt.figure(4, figsize=(6, 4))
fig4.suptitle('')
ax1 = fig4.subplots(1, 1)
ax1.plot(tr_sort, tr_cdf, 'b', label="Numerische Auswertung")
ax1.axis([110, 150, 0, 1])
ax1.set_xlabel(r'Temperatr $T_R$ / °C')
ax1.set_ylabel(r'Wahrscheinlichkeitsverteilung $F(T_R)$ / 1/°C')
ax1.grid(True)
print("Annahme des normalverteilten Luftmassenstroms gerechtfertigt.")


""" Aufgabenteil i: Vergleich der Empfindlichkeiten """

# Tolerance causes and their impact
simulation = pd.DataFrame({'iu': iu_sim.reshape(-1),
                           'ir': ir_sim.reshape(-1),
                           'ru': ru_sim.reshape(-1),
                           'rr': rr_sim.reshape(-1),
                           'r_alpha': r_alpha_sim.reshape(-1),
                           'tr': tr_sim.reshape(-1)})
poly = ols("tr ~ iu", simulation)
model = poly.fit()
e_iu_sim = model.params['iu']
poly = ols("tr ~ ir", simulation)
model = poly.fit()
e_ir_sim = model.params['ir']
poly = ols("tr ~ ru", simulation)
model = poly.fit()
e_ru_sim = model.params['ru']
poly = ols("tr ~ rr", simulation)
model = poly.fit()
e_rr_sim = model.params['rr']
poly = ols("tr ~ r_alpha", simulation)
model = poly.fit()
e_r_alpha_sim = model.params['r_alpha']

# collect analytical calculated and simulated sensitivities
labels = ['I_U', 'I_R', 'R_U', 'R_R', 'R_APLHA']
e_ana = np.log10(np.abs(np.append(e_iu, [e_ir, e_ru, e_rr, e_r_alpha])))
e_sim = np.log10(np.abs(np.append(e_iu_sim, [e_ir_sim, e_ru_sim, e_rr_sim,
                                             e_r_alpha_sim])))
index = np.arange(len(labels))
width = 0.35

# Visualization
fig5 = plt.figure(5, figsize=(6, 4))
fig5.suptitle('')
ax1 = fig5.subplots(1, 1)
rects1 = ax1.bar(index - width/2, e_ana, width, color='b', label='Rechnung')
rects2 = ax1.bar(index + width/2, e_sim, width, color='r', label='Simulation')
ax1.axis([-1, 5, -5, 10])
ax1.set_xticks(index)
ax1.set_xticklabels(labels)
ax1.set_xlabel(r'Beitrag')
ax1.set_ylabel(r'Empfindichkeit $log_{10}(E)$')
ax1.grid(True)
ax1.legend(loc=9, ncol=2)
print()
print("Empfindlichkeiten wegen der Korrelation der Widerstände nicht gleich.")
