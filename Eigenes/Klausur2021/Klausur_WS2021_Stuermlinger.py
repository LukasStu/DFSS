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


# Wahrscheinlichkeitsverteilung
fig2, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(data10, density='true')
ax1.set_xlabel(r'c/ m/s bei 10°C')
ax1.set_ylabel(r'Wahrscheinlichkeit')
ax2.hist(data20, density='true')
ax2.set_xlabel(r'c/ m/s bei 10°C')
ax2.set_ylabel(r'Wahrscheinlichkeit')

# Dichtefunktion plotten
xaxes10 = np.arange(min(data10), max(data10) ,1E-3)
xaxes20 = np.arange(min(data20), max(data20) ,1E-3)
f10 = norm.pdf(xaxes10, loc=c10_quer, scale=s_c10)
f20 = norm.pdf(xaxes20, loc=c20_quer, scale=s_c20)
ax1.plot(xaxes10, f10, 'r')
ax2.plot(xaxes20, f20, 'r')


"""c) Konfidenzbereiche"""
""" t-Verteilung bei Mittelwerte, Chi² bei Varianz"""
 #data10 Mittelwert
gamma95 = 0.95
c1_95 = t.ppf((1-gamma95)/2,N10-1)
c2_95 = t.ppf((1+gamma95)/2,N10-1)
c10_min_95 = c10_quer-c2_95*s_c10/np.sqrt(N10)
c10_max_95 = c10_quer-c1_95*s_c10/np.sqrt(N10)
print(' ')
print('Untere Grenze für \u03bc_c(10°C): {:.4f}m/s'.format(c10_min_95))
print('Obere Grenze für \u03bc_c(10°C):{:.4f}m/s'.format(c10_max_95))

#data20 Mittelwert
c1_95 = t.ppf((1-gamma95)/2,N20-1)
c2_95 = t.ppf((1+gamma95)/2,N20-1)
c20_min_95 = c20_quer-c2_95*s_c20/np.sqrt(N20)
c20_max_95 = c20_quer-c1_95*s_c20/np.sqrt(N20)
print(' ')
print('Untere Grenze für \u03bc_c(20°C): {:.4f}m/s'.format(c20_min_95))
print('Obere Grenze für \u03bc_c(20°C):{:.4f}m/s'.format(c20_max_95))

#data10 Varianz
# Grenzen für 95%
c1_95 = chi2.ppf((1-gamma95)/2,N10-1)
c2_95 = chi2.ppf((1+gamma95)/2,N10-1)
# 95% Konfidenzintervall
c10_sigma_min = np.sqrt((N10-1)/c2_95)*s_c10
c10_sigma_max = np.sqrt((N10-1)/c1_95)*s_c10
print(' ')
print('Untere Grenze für sig_c(10°C): {:.4f}m/s'.format(c10_sigma_min))
print('Obere Grenze für sig_c(10°C):{:.4f}m/s'.format(c10_sigma_max))

#data20 Varianz
# Grenzen für 95%
c1_95 = chi2.ppf((1-gamma95)/2,N20-1)
c2_95 = chi2.ppf((1+gamma95)/2,N20-1)
# 95% Konfidenzintervall
c20_sigma_min = np.sqrt((N20-1)/c2_95)*s_c20
c20_sigma_max = np.sqrt((N20-1)/c1_95)*s_c20
print(' ')
print('Untere Grenze für sig_c(20°C): {:.4f}m/s'.format(c20_sigma_min))
print('Obere Grenze für sig_c(20°C):{:.4f}m/s'.format(c20_sigma_max))


"""d) Prüfung auf gleiche Mittelwerte"""
s_gesamt = np.sqrt((s_c10**2*(N10-1)+s_c20**2*(N20-1))/(N10+N20-2))

# Intervallgrenzen der t-verteilen ZV mit NA+NB-2 Freiheitsgraden
C = t.interval(gamma95,N10+N20-2)
# Annahmebereich berechnen
Annnahme = np.array([C[0]*np.sqrt(1/N10+1/N20)*s_gesamt,C[1]*np.sqrt(1/N10+1/N20)*s_gesamt])

# Vergleich mit Stichprobe
if (c10_quer-c20_quer)<Annnahme[0] or (c10_quer-c20_quer)>=Annnahme[1]:
    print("Hypothese verworfen. Die Mittelwerte weichen signifikant voneinander ab")
else:
    print("Hypothese angenommen")
    print("{:.4f} < {:.4f} <= {:.4f}".format(Annnahme[0],(c10_quer-c20_quer),[1]))
    
    



"""Regression"""
"""e) Regressionskoeffizienten minimieren, graphisch darstellen"""

data = loadmat('Kompensation')['kompensation']
regress = pd.DataFrame({'rH' : np.reshape(data[:,0],-1),
                        'theta' : np.reshape(data[:,1],-1),
                        'c' : np.reshape(data[:,2],-1)})

model = ols("c ~ rH + theta + I(rH*theta) + I(rH**2) + I(theta**2)", regress).fit()
print('rH**2 entfernt wegen hohem p-Value ')
model = ols("c ~ rH + theta + I(rH*theta) + I(theta**2)", regress).fit()
print('theta**2 entfernt wegen hohem p-Value ')
model = ols("c ~ rH + theta + I(rH*theta)", regress).fit()
print('Wechselterm entfernt wegen p-Value>5% ')
model = ols("c ~ rH + theta", regress).fit()
print('rH entfernt wegen p-Value>5% ')
model = ols("c ~ theta", regress).fit()
print(model.summary())
print('c ist nur von theta linear abhängig')
b = model.params

""" Darstellung der Regressionsfunktion nach Reduktion """

theta_plot = np.arange(0, 40, 0.01)
cplot = b[0] + b[1]*theta_plot
fig3, ax = plt.subplots(1,1)
ax.plot(theta_plot, cplot)
ax.set_xlabel(r'theta/°C')
ax.set_ylabel(r'c/ m/s')

"""f) Konfidenzbereich signifikante Regressionskoeff"""
st, data, ss2 = summary_table(model, alpha=0.05)





"""Statistische Tolerierung"""
"""a) Grenzwertsatz"""
# Einflussgrößen
hR0 = 3
T_hR = 1E-3
sig_hR = T_hR/6

c0 = 343.2
T_c = 0.1
sig_c = T_c/6

T0 = 8.7413E-3
T_T = 10E-6
T_min = T0 - T_T/2
T_max = T0 + T_T/2
sig_T = T_T/np.sqrt(12)
   
# Symbolisch berechnen
hR_sym, c_sym, T_sym = syms.symbols('hR_sym, c_sym, T_sym')
h_sym = hR_sym -(c_sym*T_sym)/2

# Empfindlichkeiten symbolisch berechnen
E_hR = h_sym.diff(hR_sym)
E_c = h_sym.diff(c_sym)
E_T = h_sym.diff(T_sym)


# Werte berechnen
# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {hR_sym:hR0, c_sym:c0, T_sym:T0}
E_hR =  float(E_hR.evalf(subs=values))
E_c =  float(E_c.evalf(subs=values))
E_T =  float(E_T.evalf(subs=values))

sig_c_gr = np.sqrt((E_hR*sig_hR)**2 +
                   (E_c*sig_c)**2 +
                   (E_T*sig_T)**2)

GAMMA = 0.95
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
tol_c_grenz = (c2 - c1) * (E_hR*sig_hR)**2
print('Toleranz mit Grenzwertsatz (m): ',tol_c_grenz)

"""b) Simulation"""

# Generieren von Zufallszahlen
N = 10000
hRsim = np.random.normal(hR0, sig_hR, N)
csim = np.random.normal(c0, sig_c, N)
Tsim = np.random.uniform(T0-T_T/2, T0+T_T/2, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
csim = hRsim-(csim*Tsim)/2


""" Statistische Simulation und Auswertung als Häufigkeitsverteilung """

c_sort = np.sort(csim)
n_cdf = np.arange(1,N+1,1)/N

index_min = np.min(np.where(n_cdf >= (1-GAMMA)/2))
index_max = np.min(np.where(n_cdf >= (1+GAMMA)/2))
n_tol_num = c_sort[index_max] - c_sort[index_min]
print("Toleranz bei numerischer Simulation =", round(n_tol_num, 4))             
        

"""leider nicht fertig geworden, deshlab auskommentiert"""
# """i) Faltung""" 
# """ Definition  Wahrscheinlichkeitsdichten """

# N_RES = 1e-7

# n_uadc = np.arange(-0.002, 0.002+N_RES, N_RES)
# pdf_uadc = norm.pdf(n_uadc, 0, np.abs(e_uadc*uadc_sig))

# n_uoff = np.arange(-0.002, 0.002+N_RES, N_RES)
# pdf_uoff = stats.uniform.pdf(n_uadc, e_uoff*uoff_min, e_uoff*uoff_tol)

# n_uadc_uoff = np.arange(-0.004, 0.004+N_RES, N_RES)
# pdf_uadc_uoff = np.convolve(pdf_uadc, pdf_uoff)*N_RES


# n_na = np.arange(-0.002, 0.002+N_RES, N_RES)
# pdf_na = stats.uniform.pdf(n_na, e_na*na_min, e_na*na_tol)

# n_uadc_uoff_na = np.arange(-0.006, 0.006+N_RES, N_RES)
# pdf_uadc_uoff_na = np.convolve(pdf_uadc_uoff, pdf_na)*N_RES

# cdf_uadc_uoff_na = np.cumsum(pdf_uadc_uoff_na)*N_RES
# cdf_uadc_uoff_na = cdf_uadc_uoff_na / np.max(cdf_uadc_uoff_na)

# index_min = np.min(np.where(cdf_uadc_uoff_na >= (1-GAMMA)/2))
# index_max = np.min(np.where(cdf_uadc_uoff_na >= (1+GAMMA)/2))
# n_tol_con = n_uadc_uoff_na[index_max] - n_uadc_uoff_na[index_min]

# print ()
# print("Toleranz bei Faltung =", round(n_tol_con/e_q, 4))






"""Faltung ungegenau, wegen nur 3 Größen. Numerische Simulation
ist genau, weil keine bestimmte Vertielung angenommen. Grenzwertsatz
ist ungenau, weil nur 3 Größen"""





"""MSA"""

"""k) Cg, Cgk"""
# Systematic Deviation and Repeatability

# Load data set and reference
Y_TOLERANCE = 0.01
Y_REPEAT_REFERENCE = 1.5
data = loadmat('SystematischerMessfehler')['h']
y_repeat_test = data.reshape(-1)
y_repeat_len = np.size(y_repeat_test)

# Visualization
fig6 = plt.figure(1, figsize=(6, 4))
fig6.suptitle('')
ax = fig1.subplots(1, 1)
ax.plot(np.arange(0, y_repeat_len)+1, y_repeat_test, 'bo-')
ax.plot(np.arange(0, y_repeat_len)+1,
         Y_REPEAT_REFERENCE*np.ones(y_repeat_len), 'r')
ax.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE+0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
ax.plot(np.arange(0, y_repeat_len)+1,
         (Y_REPEAT_REFERENCE-0.1*Y_TOLERANCE)*np.ones(y_repeat_len), 'g--')
#ax1.axis([0, 51, 18.1, 18.5])
ax.set_xlabel('Messung')
ax.set_ylabel('Temperatur $T$ / °C')
ax.set_title('Visualisierung der systematischen Messabweichung')
ax.grid(True)

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

print('Cg bewertet Streuung in Relation zut Toleranz')
print('Cgk berücksichtigt die systematische Abweichung')
print('Das Messsystem hat einen systematischen Fehler')



# Linearity

# Load data set and reference
data = loadmat('Linearitaet')['linearitaet']
y_linearity = pd.DataFrame({'reference': np.repeat([1.1, 1.3,
                                                    1.5, 1.7, 1.9], 10),
                            'value': data.reshape(-1, order='F')})
y_linearity["deviation"] = y_linearity["value"] - y_linearity["reference"]

# Visualization
fig2 = plt.figure(2, figsize=(12, 4))
fig2.suptitle('')
ax1, ax2 = fig2.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
ax1.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax1.axis([0, 60, -0.1, 0.1])
ax1.set_xlabel('Referenzwert $T$ / °C')
ax1.set_ylabel(r' Abweichung $\Delta T$ / °C')
ax1.set_title('Bewertung des Konfidenzbereichs')
ax1.grid(True)
ax2.plot(y_linearity["reference"], y_linearity["deviation"], 'b+')
#ax2.axis([0, 60, -0.1, 0.1])
ax2.set_xlabel('Referenzwert $T$ / °C')
ax2.set_ylabel(r' Abweichung $\Delta T$ / °C')
ax2.set_title('Mittelwerte zur Lineartätsbewertung')
ax2.grid(True)

# Regression function with confidence bounds
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
print(model.summary())
y_plot = np.arange(0, 2, 1E-3)
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
    
    
    
"""m) hier muss ein Tippfehler sein. Deshalb ist es auskommentiert"""
"""Bitte bewerten"""
"""Außerdem scheine ich mit den Grafikvariablen durcheinander gekommen zu sein..."""
# Assessment of process variation according to prodedure 3

# Load, format and evaluate data
# data = loadmat('Streuverhalten')
# y_variation_3 = pd.DataFrame({'Part': np.tile(np.arange(0, 25, 1), 2),
#                               'Measurement': np.repeat([1, 2], 25),
#                               'Value': data['h'].reshape(-1, order='F')})
# Y_K = 25
# Y_N = 2


# # Calculation of normalized squares sums making use of anova table
# model = ols('Value ~ C(Part)', data=y_variation_3).fit()
# anova1 = sm.stats.anova_lm(model, typ=2)
# anova1["M"] = anova1["sum_sq"]/anova1["df"]

# # estimations of variance and calculation of GRR and ndc
# equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
# part_variation = np.sqrt((anova1.loc["C(Part)", "M"]
#                           - anova1.loc["Residual", "M"])/Y_N)
# grr = equipment_variation
# grr_relative = 6*grr/Y_TOLERANCE
# ndc = 1.41*part_variation/grr
# print("")
# print("")
# print("Streuverhalten: Verfahren 3")
# print("")
# print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
# print("Number of Distict Categories ndc = ", round(ndc, 3))

# # Visualization
# y_variation_3_multi\
#     = y_variation_3.set_index(['Measurement', 'Part'])
# fig4 = plt.figure(4, figsize=(12, 4))
# fig4.suptitle('')
# ax1, ax2 = fig4.subplots(1, 2)
# ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],
#          'b', label='Messung 1')
# ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],
#          'r:', label='Messung 2')
# #ax1.axis([0, 26, 0.9, 1.1])
# ax1.set_xlabel('Stichprobe')
# ax1.set_ylabel(r'Höhe/m')
# ax1.set_title('Streuverhalten')
# ax1.grid(True)
# ax1.legend(loc=9, ncol=3)






