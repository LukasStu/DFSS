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


"""Hypothesentest auf gleiche Mittelwerte"""
# Da Varianz unbekannt, t-Variable
"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data1 = loadmat('Signifikanz')['m1']
data2 = loadmat('Signifikanz')['m2']
x1 = data1.reshape(-1)
x2 = data2.reshape(-1)

N = x1.size
M = x2.size
x1_quer = np.mean(x1)
x2_quer = np.mean(x2)
s1 = np.std(x1,ddof=1)
s2 = np.std(x2,ddof=1)
# Anmerkung: s1² und s2² sind chi²-verteilt. Die Summe ist ebenfalls
# chi²-verteilt, mit Summe der Freiheitsgrade: N-1+M-1 = N+M-2
s_gesamt = np.sqrt((s1**2*(N-1)+s2**2*(M-1))/(N+M-2))

"Konfidenzintervall"
gamma = 0.95
c = stats.t.interval(gamma,N+M-2)
delta_mu_min = (x1_quer-x2_quer)-c[1]*np.sqrt(1/N+1/M)*s_gesamt
delta_mu_max = (x1_quer-x2_quer)-c[0]*np.sqrt(1/N+1/M)*s_gesamt
print(' ')
print('Untere Grenze für delta mu: ', delta_mu_min)
print('Obere Grenze für delta mu: ', delta_mu_max)
print('Da die Null innerhalb des Konfidenzintervalls liegt,')
print('kann die Nullhypothese aufrechterhalten werden:')
print('Der Einfluss der Drehzahl n hat keinen sign. Einfluss')
print('auf die Masse m')



"""b) ANOVA-Tabelle, Varianzanalyse"""
cat = np.concatenate(([data1, data2]))
df = pd.DataFrame({'n': np.repeat([75, 128],8),
                    'm': cat.reshape(-1)})

""" ANOVA durchführen, dazu Modell aufbauen"""

model = ols('m ~ C(n)', data=df).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
pVal = anova1.loc['C(n)','PR(>F)']

# Bewertung des p-Values
if pVal > 0.05:
    print('-> Die Masse hängt nicht der Drehzahl ab\n')
else:
    print('-> Die Masse hängt von der Drehzahl ab\n')
print('Der Variablen liegt eine F-Verteilung zu Grunde')




"""c) lineare Regression"""
poly = ols('m ~ n', df)
model = poly.fit()
#plt.plot(df['n'], df['m'],'b+')
pVal = model.pvalues['n']

# Bewertung des p-Values des linearen Koeffs
if pVal > 0.05:
    print('->lin. Koeff ist nicht signifikat\n')
else:
    print('->lin. Koeff ist signifikant\n')
    
print('\nAlle Tests führen zum selben Ergebnis: Die Masse ist nicht von der Drehzahl abhängig')

"""Plausibilisierung mit Boxplot"""
df = pd.DataFrame({'m1': data1.reshape(-1),
                   'm2': data2.reshape(-1)})
#df.boxplot()


"""Statistische Tolerierung"""
"""m) Vollquadratischer Ansatz mit Minimierung der Koeffizienten"""

data1 = loadmat('Tolerierung')['TZ'].astype(np.float64).reshape(-1)
data2 = loadmat('Tolerierung')['m5'].reshape(-1)

df = pd.DataFrame({'Tz': data1,
                   'm': data2})
poly = ols('m ~ Tz + I(Tz**2)', df)
model = poly.fit()
print(model.summary())
print('\nAlle Koeffizienten sind signifikant')

# Covariance matrix of regression coefficients
beta_cov = poly.normalized_cov_params*model.mse_resid
print("")
print("Kovarianzmatrix der Regressionkoefizienten")
print("")
print(beta_cov)
print("")
print("Elemente der Kovarianzmatrix sind auch außerhalb der Hauptdiagonalen")
print("von null verschieden, Regressionskoeffizienten sind abhängig")


"""o)"""
# m = b0 + b1*T + b2*T²
# dm = 1*delta_b0 + T*delta_b1 + T²*delta_b2 +(b1+2*T*b2)*delta_T 
T0 = 230.0
Tmax = 235.0
Tmin = 225.0
TT = Tmax-Tmin
var_T = (TT/6)**2

var_b0 = beta_cov[0,0]

b0 = model.params['Intercept']
b1 = model.params['Tz']
var_b1 = beta_cov[1,1]

b2 = model.params['I(Tz ** 2)']
var_b2 = beta_cov[2,2]

var_b0b1 = beta_cov[0,1]
var_b0b2 = beta_cov[0,2]
var_b1b2 = beta_cov[1,2]

E_b0 = 1
E_b1 = T0
E_b2 = T0**2
E_T = b1+2*T0*b2


tol = 6*np.sqrt(E_b0**2*var_b0 +
                E_b1**2*var_b1 +
                E_b2**2*var_b2 +
                E_T**2*var_T +
                2*var_b0b1*E_b0*E_b1 +
                2*var_b0b2*E_b0*E_b2 +
                2*var_b1b2*E_b1*E_b2)

print('Toleranz',tol)

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


# df = pd.DataFrame({'cause': [E_b0**2*var_b0,
#                              E_b1**2*var_b1,
#                              E_b2**2*var_b2,
#                              E_T**2*var_T,
#                              2*var_b0b1*E_b0*E_b1,
#                              2*var_b0b2*E_b0*E_b2,
#                              2*var_b1b2*E_b1*E_b2]},
#                   index=['b0','b1','b2','T','b0b1','b0b2','b0b3'])
# df.plot.bar()