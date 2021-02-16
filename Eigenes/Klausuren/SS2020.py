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

"""Statistische Auswertung und Wiederholbarkeit"""
"""a) Messreihe aus Häufigkeitsdiagramm"""

data = loadmat('Durchflussmessung')['QREF']
Q_ref = np.array(data).reshape(-1)

Q_quer = np.mean(Q_ref)
s_Q = np.std(Q_ref,ddof=1)
N = np.size(Q_ref)

#absolute Häufigkeit
fig1, ax = plt.subplots()
ax.hist(Q_ref)
ax.set_xlabel(r'Durchfluss $Q_{\mathrm{ist}}/$m³/h')
ax.set_ylabel(r'Absolute Häufigkeit')
ax.grid(True)

"""Schätzung der Parameter mu und sigma"""
print('b) Parameterschätzung\n')
# mu
# Konfidenzzahlen
GAMMA = 0.95

# Grenzen für 95%
# Konfidenzintervall für t-verteilte Variable
c1 = t.ppf((1-GAMMA)/2,N-1)
c2 = t.ppf((1+GAMMA)/2,N-1)

# 95% Konfidenzintervall
mu_min = Q_quer-c2*s_Q/np.sqrt(N)
mu_max = Q_quer-c1*s_Q/np.sqrt(N)
print("< {:.4f} =< mu =< {:.4f}".format(mu_min, mu_max))

# sigma
# Grenzen für 95%
# Konfidenzintervall für Chi²-verteilte Variable
c1 = chi2.ppf((1-GAMMA)/2,N-1)
c2 = chi2.ppf((1+GAMMA)/2,N-1)
# 95% Konfidenzintervall
sigma_min = np.sqrt((N-1)/c2)*s_Q
sigma_max = np.sqrt((N-1)/c1)*s_Q
print("< {:.4f} =< sigma =< {:.4f}".format(sigma_min, sigma_max))

"""c) Wahrscheinlichkeitsdichte """
# Wahrscheinlichkeitsverteilung
fig2, ax2 = plt.subplots()
ax2.hist(Q_ref, density='true')
ax2.set_xlabel(r'Durchfluss $Q_{\mathrm{ist}}/$m³/h')
ax2.set_ylabel(r'Wahrscheinlichkeit')

axes = np.arange(min(Q_ref), max(Q_ref), 1E-5)
f_Q = norm.pdf(axes, loc=Q_quer, scale=s_Q)
ax2.plot(axes, f_Q, 'r')
ax2.grid(True)


"""d) systematischer Messfehler"""
#mu1 != mu0
alpha = 0.05
mu_0 = 0.5
c_min = t.ppf(alpha/2, N-1)
c_max = t.ppf(1-alpha/2, N-1)
Q_quer_Annahme = np.array([mu_0+c_min*s_Q/np.sqrt(N),mu_0+c_max*s_Q/np.sqrt(N)])
print("Annahme bei {:.4f} < Q_quer <= {:.4f}".format(Q_quer_Annahme[0], Q_quer_Annahme[1]))
print("Q_quer={:.4f}".format(Q_quer))
print("->systematischer Fehler")
      
# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-0.01, 0.01, num=10000)
G = 1+norm.cdf((Q_quer_Annahme[0]-(mu_0+d_mu))/(s_Q/np.sqrt(N)))-\
    norm.cdf((Q_quer_Annahme[1]-(mu_0+d_mu))/(s_Q/np.sqrt(N)))

fig3, ax3 = plt.subplots()
ax3.plot(d_mu,G,label='Gütefunktion')
ax3.set_xlabel(r'$\Delta \mu$')
ax3.set_ylabel(r'$1-\beta(\mu_1)$')
ax3.grid(True)

index = np.min(np.where(G<=0.95))
ax3.plot(d_mu[index], G[index], 'r+')

print('\nEs wird eine Abweichung von ±{:.4f}m³/h zu 95% erkannt'.format(np.abs(d_mu[index])))






print('\nStatistische Tolerierung')

"""j) Statistische Tolerierung mit Grenzwertmethode"""
# Angabe der Größen
Q0 = 0.5
TQ = 0.002
sigQ = TQ/6

pD0 = 1013
TpD = 2
pDmin = pD0 - TpD/2
pDmax = pD0 + TpD/2
sigpD = TpD/np.sqrt(12)

TD0 = 293
TTD = 1
TDmin = TD0 - TTD/2
TDmax = TD0 + TTD/2
sigTD = TTD/np.sqrt(12)

cPE0 = 5E-5
TcPE = 2.5E-6
sigcPE = TcPE/6
 
K0 = 0.95
TK = 0.005
sigK = TK/6

Tnom = 293
pnom = 1013
# Empfindlichkeiten berechnen
Q0_sym, pD0_sym, TD0_sym, cPE0_sym, K0_sym, Tnom_sym, pnom_sym\
    = syms.symbols('Q0_sym, pD0_sym, TD0_sym, cPE0_sym, K0_sym, Tnom_sym, pnom_sym')
Q_korr_sym = (pD0_sym*Tnom_sym)/(pnom_sym*TD0_sym*K0_sym)*(1+cPE0_sym*(pD0_sym-pnom_sym))*Q0_sym

E_Q = Q_korr_sym.diff(Q0_sym)
E_pD = Q_korr_sym.diff(pD0_sym)
E_TD = Q_korr_sym.diff(TD0_sym)
E_cPE = Q_korr_sym.diff(cPE0_sym)
E_K = Q_korr_sym.diff(K0_sym)

values = {Q0_sym:Q0, pD0_sym:pD0, TD0_sym:TD0, cPE0_sym:cPE0, K0_sym:K0, Tnom_sym:Tnom, pnom_sym:pnom}
EQ =  float(E_Q.evalf(subs=values))
EpD =  float(E_pD.evalf(subs=values))
ETD =  float(E_TD.evalf(subs=values))
EcPE =  float(E_cPE.evalf(subs=values))
EK =  float(E_K.evalf(subs=values))

sig_Q_korr = np.sqrt((EQ*sigQ)**2 +
                     (EpD*sigpD)**2 +
                     (ETD*sigTD)**2 +
                     (EcPE*sigcPE)**2+
                     (EK*sigK)**2)

GAMMA = 0.9973
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
tol_Q_korr = (c2 - c1) * sig_Q_korr
print("Toleranz bei Grenzwertmethode ist ±{:.4f}m³/h".format(tol_Q_korr))



"""k) Toleranzursachen"""
df = pd.DataFrame({'Toleranzbeitrag':[(EQ*sigQ)**2,
                             (EpD*sigpD)**2,
                             (ETD*sigTD)**2,
                             (EcPE*sigcPE)**2,
                             (EK*sigK)**2]},
                             index=['Q','pD','TD','cPE','K'])
df=df.sort_values('Toleranzbeitrag')
df.plot.bar()




"""l) Verifikation mit stat. Simulation"""
# Generieren von Zufallszahlen
N = 10000
Qnom_sim = np.random.normal(Q0, sigQ, N)
pD_sim = np.random.uniform(pDmin, pDmax, N)
TD_sim = np.random.uniform(TDmin, TDmax, N)
cPE_sim = np.random.normal(cPE0, sigcPE, N)
K_sim = np.random.normal(K0, sigK, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
Qkor_sim = pD_sim*Tnom/pnom/TD_sim/K_sim*(1+cPE_sim*(pD_sim-pnom))*Qnom_sim

# Toleranz als Prognoseintervall
Qkormean = np.mean(Qkor_sim)
Qkorstd = np.std(Qkor_sim,ddof=1)
c1 = t.ppf((1-GAMMA)/2,N-1)
c2 = t.ppf((1+GAMMA)/2,N-1)
TQkorMC = Qkorstd*np.sqrt(1+1/N)*(c2-c1)
print(' ')
print('Toleranz bei Monte-Carlo-Simulation ist ±{:.4f}m³/h'.format(TQkorMC))
Qkorr_min = Qkormean+c1*np.sqrt(1+1/N)*Qkorstd
Qkorr_max = Qkormean+c2*np.sqrt(1+1/N)*Qkorstd

print('{:.4f}'.format(Qkorr_max-Qkorr_min))


