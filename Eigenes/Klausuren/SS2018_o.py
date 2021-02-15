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

print('\nToleranz',tol)

# Zur Kontrolle symbolisch berechnen
m_sym, T_sym, b0_sym, b1_sym, b2_sym\
    = syms.symbols('m_sym, T_sym, b0_sym, b1_sym, b2_sym')
m_sym = b0_sym + T_sym*b1_sym + T_sym**2*b2_sym

# Empfindlichkeiten symbolisch berechnen
E_b0 = m_sym.diff(b0_sym)
E_b1 = m_sym.diff(b1_sym)
E_b2 = m_sym.diff(b2_sym)
E_T = m_sym.diff(T_sym)
#print(E_T)

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

print('\nToleranz',tol)





df = pd.DataFrame({'cause': [E_b0**2*var_b0,
                              E_b1**2*var_b1,
                              E_b2**2*var_b2,
                              E_T**2*var_T,
                              2*var_b0b1*E_b0*E_b1,
                              2*var_b0b2*E_b0*E_b2,
                              2*var_b1b2*E_b1*E_b2]},
                  index=['b0','b1','b2','T','b0b1','b0b2','b0b3'])
df.plot.bar()