

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
from scipy.stats import t, norm, chi2, f, uniform
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import summary_table
import sympy as syms
from sympy.core import evalf

"""Bestimmen Sie die linearisierte Maßkette delta_N im Arbeitspunkt Q0 = 0.5 m³/s unter Berücksichtigung
der oben aufgeführten Toleranzursachen. Berechnen Sie die Empfindlichkeiten EUADC, ENA
und EUOFF."""

# Angabe der Größen
U_ADC_0 = 5
T_U_ADC = 0.25
U_ADC_min = U_ADC_0 - T_U_ADC
U_ADC_max = U_ADC_0 + T_U_ADC
U_ADC_sigma = T_U_ADC/3
d_U_ADC = 2*T_U_ADC

U_OFF_0 = 10E-3
T_U_OFF = 10E-3
U_OFF_min = U_OFF_0 - T_U_OFF
U_OFF_max = U_OFF_0 + T_U_OFF
dU_OFF = 2*T_U_OFF

NA_0 = 0
T_NA = 1/2048
NA_0_min = NA_0 - T_NA
NA_0_max = NA_0 + T_NA
d_NA = 2*T_NA

k = 0.2
m = 0.04
Q_0 = 0.5


# Masskette definieren und Empfindlichkeiten berechnen
# Linearisierte Maßkette bei Q_0 = 0.5
# Variablen und Funktion symbolisch definieren
N_sym, U_ADC_0_sym, U_OFF_0_sym, NA_0_sym, k_sym, m_sym, Q_0_sym = syms.symbols('N_sym, U_ADC_0_sym, U_OFF_0_sym, NA_0_sym, k, m, Q_0')
N_sym = (m_sym*Q_0**2/k_sym**2)+(m_sym*Q_0**2/k_sym**2)*U_OFF_0_sym/U_ADC_0_sym+NA_0_sym

# Empfindlichkeiten symbolisch berechnen
E_U_ADC_sym = N_sym.diff(U_ADC_0_sym)
E_NA_sym = N_sym.diff(NA_0_sym)
E_U_OFF_sym = N_sym.diff(U_OFF_0_sym)

# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {U_ADC_0_sym:U_ADC_0, U_OFF_0_sym:U_OFF_0, NA_0_sym:NA_0, k_sym:k, m_sym:m, Q_0_sym:Q_0}
E_U_ADC =  float(E_U_ADC_sym.evalf(subs=values))
E_U_OFF = float(E_U_OFF_sym.evalf(subs=values))
E_NA = float(E_NA_sym.evalf(subs=values))

print('Empfindlichkeiten Symbolisch')
print('E_U_ADC: {:.4f}'.format(E_U_ADC))
print('E_NA: {:.4f}'.format(E_NA))
print('E_U_OFF: {:.4f}'.format(E_U_OFF))


"""c) Verifizieren Sie die berechneten Empfindlichkeiten über eine statistische Simulation."""
# Linearer Regressionskoeffizient entspricht Empfindlichkeit

N = 10000
U_ADC_sim = np.random.normal(U_ADC_0, U_ADC_sigma, N)
U_OFF_sim = np.random.uniform(U_OFF_min, U_OFF_max, N)
NA_sim = np.random.uniform(NA_0_min, NA_0_max, N)

# Berechung des Durchflusses
N_sim = (m*Q_0**2/k**2)+(m*Q_0**2/k**2)*U_OFF_sim/U_ADC_sim+NA_sim

""" Test der Empfindlichkeiten """
regress = pd.DataFrame({'U_ADC': U_ADC_sim.reshape(-1),
                        'U_OFF': U_OFF_sim.reshape(-1),
                        'NA': NA_sim.reshape(-1),
                        'N': N_sim.reshape(-1)})

poly = ols("N ~ U_ADC + U_OFF + NA" , regress)
model = poly.fit()

print('\nEmpfindlichkeiten Simulation')
print("E_U_ADC_sim:"+"{:10.4f}".format(model.params.U_ADC))
print("E_NA_sim:"+"{:10.4f}".format(model.params.NA))
print("E_U_OFF_sim:"+"{:10.4f}".format(model.params.U_OFF))









