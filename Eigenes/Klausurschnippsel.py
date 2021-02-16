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
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
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

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('<Dateiname>')['data']
X = np.array(data).reshape(-1)

x_quer = np.mean(x)
s = np.std(x,ddof=1)
N = np.size(x)

print(' ')
print('Mittelwert x: ', x_quer)
print('Standardabweichung s: ', s)


"""https://matplotlib.org/3.1.1/tutorials/text/mathtext.html"""
" Histogramme"
test=np.array([1,2,2,3,3,3,4,4,4,4])
#absolute Häufigkeit
fig1, ax = plt.subplots(1,1)
ax1.hist(test)
ax1.set_xlabel(r'')
ax1.set_ylabel(r'Absolute Häufigkeit')
# relative Häufigkeit
fig2, ax = plt.subplots(1,1)
ax1.hist(test, weights=np.zeros_like(test) + 1. / test.size)
ax1.set_xlabel(r'')
ax1.set_ylabel(r'Relative Häufigkeit')
# Wahrscheinlichkeitsverteilung
fig3, ax = plt.subplots(1,1)
ax1.hist(test, density='true')
ax1.set_xlabel(r'')
ax1.set_ylabel(r'Wahrscheinlichkeit')

"""Hypothesentest Univariat"""
"""Varianz bekannt -> ZV ist Standardnormalverteilt, wenn H0 gilt"""
#mu1 > mu0
c_max = norm.ppf(1-alpha)
x_quer_max = mu0+(c_max*sigma)/np.sqrt(N)
print("Annahme H0 bei x_quer < {:.4f}".format(x_quer_max))
# Gütefunktion für mu_1 > mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1-norm.cdf((x_quer_min-(mu_0+d_mu))/(sigma/np.sqrt(N)))

#mu1 < mu0
c_min = norm.ppf(alpha)
x_quer_max = mu0+(c_max*sigma)/np.sqrt(N)
print("Annahme bei x_quer > {:.4f}".format(x_quer_min)
# Gütefunktion für mu_1 < mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = norm.cdf((x_quer_max-(mu_0+d_mu))/(sigma/np.sqrt(N)))

      
#mu1 != mu0
c_min = norm.ppf(alpha/2)
c_max = norm.ppf(1-alpha/2)
x_quer_Annahme = np.array([mu_0+c_min*sigma/np.sqrt(N),mu_0+c_max*sigma/np.sqrt(N)])
print("Annahme bei {:.4f} < x_quer <= {:.4f}".format(x_quer_Annahme[0],x_quer_Annahme[1])
# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1+norm.cdf((x_quer_Annahme[0]-(mu_0+d_mu))/(sigma/np.sqrt(N)))-norm.cdf((x_quer_Annahme[1]-(mu_0+d_mu))/(sigma/np.sqrt(N)))

fig, ax = plt.subplots()
ax.plot(d_mu,G,label='Gütefunktion')
ax.set_xlabel(r'$\Delta \mu$')
ax.set_ylabel(r'$1-\beta(\mu_1)$')
ax.grid(True)

# Numerische Auswertung
# Abweichung gesucht
index = np.min(np.where(G<=0.95))
ax3.plot(d_mu[index], G[index], 'r+')
print('\nEs wird eine Abweichung von ±{:.4f}m³/h zu 95% erkannt'.format(np.abs(d_mu[index])))

# Wahrscheinlichkeit gesucht
index = np.min(np.where(d_mu>=-0.0047))
ax3.plot(d_mu[index], G[index], 'b+')
print('\nEs wird eine Abweichung von ±0.0047m³/h zu {:.2%} erkannt'.format(np.abs(G[index])))



""" Korrelationsanalyse """
""" H_0: s_alpha² = 0 """
""" H_0 = Es gibt keinen signifikaten Einfluss zwischen den Gruppen"""


"""ANOVA"""
model = ols('m ~ C(n)', data=mass_ols).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
print("")
print("Anova auf Abweichung mit p-value = ",
      round(float(anova1["PR(>F)"]["C(n)"]), 4))

if anova1["PR(>F)"]["C(n)"] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")
    

"""Regression"""
poly = ols("m ~ n", mass_ols)
model = poly.fit()
print("")
print("Prüfung Signifikanz des linearen Terms mit p-value = ",
      round(float(model.pvalues["n"]), 4))
if model.pvalues["n"] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")
    
"""Statistische Simulation"""
# Zur Kontrolle symbolisch berechnen
m_sym, T_sym, b0_sym, b1_sym, b2_sym\
    = syms.symbols('m_sym, T_sym, b0_sym, b1_sym, b2_sym')
m_sym = b0_sym + T_sym*b1_sym + T_sym**2*b2_sym

# Empfindlichkeiten symbolisch berechnen
E_b0 = m_sym.diff(b0_sym)
E_b1 = m_sym.diff(b1_sym)
E_b2 = m_sym.diff(b2_sym)
E_T = m_sym.diff(T_sym)
print(E_T)

# Werte berechnen
# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {T_sym:T0, b0_sym:b0, b1_sym:b1, b2_sym:b2}
E_b0 =  float(E_b0.evalf(subs=values))
E_b1 =  float(E_b1.evalf(subs=values))
E_b2 =  float(E_b2.evalf(subs=values))
E_T =  float(E_T.evalf(subs=values))

tol = 6*np.sqrt(E_b0**2*var_b0 +
                E_b1**2*var_b1 +
                E_b2**2*var_b2 +
                E_T**2*var_T +
                2*var_b0b1*E_b0*E_b1 +
                2*var_b0b2*E_b0*E_b2 +
                2*var_b1b2*E_b1*E_b2)

print('Toleranz',tol)

"""Alternativ über Prognosebereich"""
GAMMA = 0.9973
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
tol_Q_korr = (c2 - c1) * sig_Zielgröße

""" Statistische Simulation für unkorrlierte Spannungen """

# Generieren von Zufallszahlen
N = 10000
Qnom_sim = np.random.normal(Qnom_0, sigQnom, N)
pD_sim = np.random.uniform(pD_min, pD_max, N)
TD_sim = np.random.uniform(TD_min, TD_max, N)
cPE_sim = np.random.normal(cPE_0, sigcPE, N)
K_sim = np.random.normal(K_0, sigK, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
Qkor_sim = pD_sim*TNOM/PNOM/TD_sim/K_sim*(1+cPE_sim*(pD_sim-PNOM))*Qnom_sim

# Toleranz als Prognoseintervall
Qkormean = np.mean(Qkor_sim)
Qkorstd = np.std(Qkor_sim,ddof=1)
gamma = 0.9973
c1 = t.ppf((1-gamma)/2,N-1)
c2 = t.ppf((1+gamma)/2,N-1)
TQkorMC = Qkorstd*np.sqrt(1+1/N)*(c2-c1)
Qkorr_min = Qkormean+c1*np.sqrt(1+1/N)*Qkorstd
Qkorr_max = Qkormean+c2*np.sqrt(1+1/N)*Qkorstd
print(' ')
print('Statistische Tolerierung über Monte-Carlo-Simulation: ', round(TQkorMC,3))

    