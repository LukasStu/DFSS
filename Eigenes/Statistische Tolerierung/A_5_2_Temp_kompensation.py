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
from scipy import constants

# Einflussgrößen mit Toleranzangabe
# Stahl
alpha_s0 = 11.8E-6
alpha_s_min = alpha_s0*0.95
alpha_s_max = alpha_s0*1.05
T_alpha_s = alpha_s_max - alpha_s_min
dT_alpha_s = 0.1
sig_alpha_s = dT_alpha_s/6

# Messing
alpha_m0 = 22.1E-6
alpha_m_min = alpha_m0*0.90
alpha_m_max = alpha_m0*1.10
T_alpha_m = alpha_m_max - alpha_m_min
dT_alpha_m = 0.2
"""a) Stellen Sie eine Maßkette für die Pendellänge L mit dem Maßen A, B und C auf."""
# L = A-B+C



"""b) Welche Länge muss das Pendel für eine Periodendauer von T0 = 1 s haben?"""
L = (1/2/constants.pi)**2*9.81
print('Länge für T_0 = 1s: {:.4f}m'.format(L))



"""c) Wie dimensionieren Sie die Längen (A + C) und B, damit die Pendellänge
bei Einhalten der Soll-werte für alpha_S und alpha_M von der Temperatur unabhängig ist
und eine Periodendauer von T0 = 1 s auf-weist?"""
# L_0 = (A+B)*(1+alpha_s*dT)-C*(1+alpha_m*dT)
A_C = L/(1-(alpha_s0/alpha_m0))
print('Länge für A+C: {:.4f}m'.format(A_C))

B = A_C *alpha_s0/alpha_m0
print('Länge für B: {:.4f}m'.format(B))


"""d) Geben Sie eine Gleichung zur Berechnung der Abweichung dT
bei beliebigen Abweichungen thermischen Ausdehnungskoeffizienten
dalphaS und daphaM und einer festen Temperatur T - T0 = 10 K an."""


# Variablen und Funktion symbolisch definieren
A_sym, C_sym, B_sym, alpha_m_sym, alpha_s_sym, g_sym, pi_sym, dtheta_sym = \
    syms.symbols('A_sym, C_sym, B_sym, alpha_m_sym, alpha_s_sym, g_sym, pi_sym, dtheta_sym')   
T_sym = 2*pi_sym*((A_sym+C_sym*(1+alpha_s_sym*dtheta_sym)-B_sym*(1+alpha_m_sym*dtheta_sym)/g_sym))**(1/2)

# Empfindlichkeiten symbolisch berechnen
E_m_sym = T_sym.diff(alpha_m_sym)
E_s_sym = T_sym.diff(alpha_s_sym)

"""e) Berechnen Sie die Toleranz von der Schwingungsdauer T bei einer
Temperatur theta - theta0 = 10 K mithilfe des Grenzwertsatzes der Wahrscheinlichkeit.
Geben Sie ein 99.73%-Konfidenzbereich für die Schwingungsdauer T an."""

# A und C festlegen
C = 0.1
A = A_C-C

# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {A_sym:A, C_sym:C, B_sym:B, alpha_m_sym:alpha_m0, alpha_s_sym:alpha_s0, g_sym:9.81, pi_sym:3.1416, dtheta_sym:10 }
E_m = float(E_m_sym.evalf(subs=values))
E_s = float(E_s_sym.evalf(subs=values))

print('E_m: {:.4f}'.format(E_m))
print('E_s: {:.4f}'.format(E_s))

""" Statistische Tolerierung mit Grenzwertsatz """
# Umrechnung Toleranzangabe in Standardabweichung 
# Achtung: Toleranangabe in Prozent, nicht absolut
sig_alpha_m = dT_alpha_m/np.sqrt(12)

""" Grenzwertmethode """
T_T = 6*np.sqrt((sig_alpha_m*E_m)**2 + (sig_alpha_s*E_s)**2)
print(' ')
print('Toleranzbereich bei Grenzwertmethode: {:.4f}s'.format(T_T))

# ODER mit Konfidenzzahl GAMMA

GAMMA = 0.9973
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
T_T2 = (c2 - c1) *\
    np.sqrt((sig_alpha_m*E_m)**2
            + (sig_alpha_s*E_s)**2)
print("")
print('Toleranz bei Grenzwertmethode: {:.4f}s'.format(T_T2))

"""f) Alternativ kann die Toleranz von der Schwingungsdauer T
bei einer Temperatur  - 0 = 10 K mithilfe der Faltung berechnet werden.
Bestimmen Sie dazu die Verteilung der Schwingungsdauer T.
Geben Sie ein 99.73%-Konfidenzbereich für die Schwingungsdauer T an."""


""" Definition der Widerstandsbereiche und ihrer Wahrscheinlichkeitsdichten """
# Definition der gemeinsamen Auflösung im Spannungsbereich
da = 1E-6;


# Normalverteilung
alpha_s_con_min = -0.1
alpha_s_con_max = 0.1
alpha_s_con = np.arange(alpha_s_con_min, alpha_s_con_max+da, da)
f1 = norm.pdf(alpha_s_con,0,np.abs(E_s*sig_alpha_s))

# Gleichverteilung
alpha_m_con_min = -0.1
alpha_m_con_max = 0.1
dalpha_m_con = np.arange(alpha_m_con_min, alpha_m_con_max+da, da)
dalpha_m_con_min = -0.05
dalpha_m_con_max = 0.05
f2 = uniform.pdf(dalpha_m_con, np.abs(dalpha_m_con_min*E_m), np.abs((dalpha_m_con_max-dalpha_m_con_min)*E_m))


""" Wahrscheinlichkeitsverteilung des Gesamtwiderstandes über Faltung """
# Faltung der beiden ersten Wahrscheinlichkeitsdchten
f12 = np.convolve(f1,f2)*da
a12min = alpha_s_con_min + alpha_m_con_min
a12max = alpha_s_con_max + alpha_m_con_max
a12 = np.arange(a12min,a12max+da,da)

# Fehlerkorrektur der Länge
a12 = a12[0:np.size(f12)]

# Berechnung Verteiungsfunktion
F12 = np.cumsum(f12)*da
F12 = F12/np.max(F12)

# Berechnung der Toleranzgrenzen über Ausfallwahrscheinlichkeiten 
indexmin = np.max(np.where(F12 <= (1-0.9973)/2))
indexmax = np.min(np.where(F12 >= (1+0.9973)/2))
amaxCon = a12[indexmax]
aminCon = a12[indexmin]
TaCon = amaxCon - aminCon
print('Test')
print('Toleranzbereich bei Faltung: ', round(TaCon,4))