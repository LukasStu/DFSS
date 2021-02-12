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

"""e) Führen Sie eine statistische Tolerierung nach dem Grenzwertsatz
      der Wahrscheinlichkeit durch. Geben Sie die Toleranz mit einem
      Konfidenzbereich von 99 % an."""

""" Einflussgrößen mit Toleranzangabe """
gamma_010 = 1
T_gamma_01 = 0.05
gamma_01_min = gamma_010-T_gamma_01/2
gamma_01_max = gamma_010+T_gamma_01/2
sig_gamma = T_gamma_01/np.sqrt(12)

alpha0 = 2.1E-2
T_0 = 25

temp10 = 25
sig_temp10 = 0.03
temp20 = 25
sig_temp20 = 0.03
roh_T = 0.8

UR10 = 0.2
sig_UR1 = 0.01E-3
UR20 = 0.2
sig_UR2 = 0.01E-3

U10 = 10
U20 = 10

R10 = 10E3
sig_R1 = 0.05
R20 = 10E3
sig_R2 = 0.05
roh_R = 0.95

""" Masskette definieren und Empfindlichkeiten berechnen"""
# Variablen und Funktion symbolisch definieren
gamma_01_sym, alpha0_sym, T0_sym, temp1_sym, temp2_sym, UR1_sym, UR2_sym, U1_sym, U2_sym, R1_sym, R2_sym = \
    syms.symbols('gamma_01_sym, alpha0_sym, T0_sym, temp1_sym, temp2_sym, UR1_sym, UR2_sym, U1_sym, U2_sym, R1_sym, R2_sym')

gamma_02_sym = gamma_01_sym*(1+alpha0_sym*(temp1_sym-T0_sym))/(1+alpha0_sym*(temp2_sym-T0_sym))*\
    (UR2_sym/UR1_sym)*((U1_sym*R1_sym)/(U2_sym*R2_sym))

# Empfindlichkeiten symbolisch berechnen
E_gamma01 = gamma_02_sym.diff(gamma_01_sym)
E_temp1 = gamma_02_sym.diff(temp1_sym)
E_temp2 = gamma_02_sym.diff(temp2_sym)
E_UR1 = gamma_02_sym.diff(UR1_sym)
E_UR2 = gamma_02_sym.diff(UR2_sym)
E_R1 = gamma_02_sym.diff(R1_sym)
E_R2 = gamma_02_sym.diff(R2_sym)

# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {gamma_01_sym:gamma_010,
          alpha0_sym:alpha0,
          T0_sym:T_0,
          temp1_sym:temp10,
          temp2_sym:temp20,
          UR1_sym:UR10,
          UR2_sym:UR20,
          U1_sym:U10,
          U2_sym:U20,
          R1_sym:R10,
          R2_sym:R20}

Egamma01 = float(E_gamma01.evalf(subs=values))
Etemp1 = float(E_temp1.evalf(subs=values))
Etemp2 = float(E_temp2.evalf(subs=values))
EUR1 = float(E_UR1.evalf(subs=values))
EUR2 = float(E_UR2.evalf(subs=values))
ER1 = float(E_R1.evalf(subs=values))
ER2 = float(E_R2.evalf(subs=values))

# Berechnung der Standardabweichung
sigma_gamma02 = np.sqrt((Egamma01*sig_gamma)**2
                        + (Etemp1*sig_temp10)**2
                        + (Etemp2*sig_temp20)**2 + 2*roh_T*sig_temp10*sig_temp20*Etemp1*Etemp2
                        + (EUR1*sig_UR1)**2
                        + (EUR2*sig_UR2)**2
                        + (ER1*sig_R1)**2
                        + (ER2*sig_R2)**2 + 2*roh_R*sig_R1*sig_R2*ER1*ER2)

# Berechnung des 99%-Prognosebereichs von delta_gamma_02 (Mittelwert und Varianz "bekannt")
GAMMA = 0.99
c1 = norm.ppf((1 - GAMMA)/2)
c2 = norm.ppf((1 + GAMMA)/2)
tol_gamma02 = (c2 - c1) * sigma_gamma02

tol_gamma02_min = gamma_010 + c1 * sigma_gamma02
tol_gamma02_max = gamma_010 + c2 * sigma_gamma02

print("Toleranz bei Grenzwertmethode =", round(tol_gamma02, 4))







"""f) Validieren Sie das Ergebnis der Rechnung mithilfe einer statistischen Simulation."""

""" Statistische Simulation und Auswertung als Prognoseintervall """
# Generieren von Zufallszahlen
N = 10000

gamma01_sim = np.random.uniform(gamma_010-T_gamma_01/2, gamma_010+T_gamma_01/2, N)
temp1_sim = np.random.normal(temp10, sig_temp10, N)
temp2_sim = np.random.normal(temp20, sig_temp20, N)
UR1_sim = np.random.normal(UR10, sig_UR1, N)
UR2_sim = np.random.normal(UR20, sig_UR2, N)
R1_sim = np.random.normal(R10, sig_R1, N)
R2_sim = np.random.normal(R20, sig_R2, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
gamma02_sim = gamma01_sim*(1+alpha0*(temp1_sim-T_0))/(1+alpha0*(temp2_sim-T_0))*\
    (UR2_sim/UR1_sim)*((U10*R1_sim)/(U20*R2_sim))

Gmean = np.mean(gamma02_sim)
Gstd = np.std(gamma02_sim,ddof=1)
Gplot = np.arange(1-0.05,1+0.05,0.001)
fsim = norm.pdf(Gplot,Gmean,Gstd)
Fsim = norm.cdf(Gplot,Gmean,Gstd)

# Toleranz als Prognoseintervall (Mittelwert und Varianz unbekannt)
c1 = t.ppf((1-GAMMA)/2,N-1)
c2 = t.ppf((1+GAMMA)/2,N-1)
TGMC1 = Gstd*np.sqrt(1+1/N)*(c2-c1)
print(' ')
print('Toleranzbereich bei Monte-Carlo-Simulation mit Prognoseintervall: ', round(TGMC1,4))

""" Grafische Darstellung der Simulation """
fig = plt.figure(3, figsize=(12, 4))
fig.suptitle('Ergebnisse der statistischen Simulation')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(gamma02_sim,'r+')
#ax1.axis([0,N,2.35,2.65])
ax1.set_xlabel('Stichprobe $n$')
ax1.set_ylabel('Ausgangsspannung $U$ / V') 
ax1.grid(True)
ax2.hist(gamma02_sim, int(np.sqrt(N)), density=True, facecolor='b')
ax2.plot(Gplot,fsim,'r')
#ax2.axis([2.35,2.65,0,25])
ax2.set_xlabel('Ausgangsspannung $U$ / V')
ax2.set_ylabel('Wahrscheinlichkeitsdichte f($U$) $\cdot$ V') 
ax2.grid(True)




"""
g) Wie wirken sich die angegebenen Korrelationen auf die Gesamttoleranz aus? Begründen
Sie Ihre Antwort."""
print('\nroh_T12: {:.4f} E_T1: {:.4f} E_T2: {:.4f}'.format(roh_T, Etemp1, Etemp2))
print('Das Produkt ist negativ -> reduziert Toleranz')

print('\nroh_R12: {:.4f} E_R1: {:.4f} E_R2: {:.4f}\n'.format(roh_R, ER1, ER2))
print('Das Produkt ist negativ, jedoch gering -> reduziert Toleranz kaum')



"""
h) Definieren Sie einen Hypothesentest, mit dem die Leitfähigkeit des entionisierten Wassers
auf gamma0 = 1 μS geprüft werden kann. Gehen Sie von einer bekannten Standardabweichung
von sigma_gamma0 = 0.005 μS und einem Signifikanzniveau von alpha = 10 % aus."""

# Berechnung der Intervallgrenzen von z
alpha = 10/100
gamma0 = 1
sigma_gamma0 = 0.005


c1 = norm.ppf(alpha/2)
c2 = norm.ppf(1-alpha/2)

# Berechnung der Eingriffsgrenzen für x_quer
x_quer_Annahme = np.array([gamma0+c1*sigma_gamma0, gamma0+c2*sigma_gamma0])
print("{:.4f} < gamma <= {:.4f}".format(x_quer_Annahme[0],x_quer_Annahme[1]))

# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-0.025,0.025, num=10000)
G = 1+norm.cdf((x_quer_Annahme[0]-(gamma0+d_mu))/\
               (sigma_gamma0))-norm.cdf((x_quer_Annahme[1]-(gamma0+d_mu))/(sigma_gamma0))

fig, ax = plt.subplots()
ax.plot(d_mu,G,label=r'$N=%d$'%N)
ax.set_xlabel(r'$\Delta \gamma$')
ax.set_ylabel(r'$1-\beta(\gamma_1)$')
ax.grid(True)

"""i) Wie groß ist die Abweichung delta_gamma, die mit einer Wahrscheinlichkeit von 99 % erkannt wird?"""
# Berechnung der Wahrscheinlichkeit für Erkennung von delta_mu = 0.5
index = np.min(np.where(G <= 0.99))
print("Eine Abweichung von ±{:.4f} wird zu 99% erkannt".format(np.abs(d_mu[index])))

# Punkt markieren
ax.plot(d_mu[index],G[index],'r+')







