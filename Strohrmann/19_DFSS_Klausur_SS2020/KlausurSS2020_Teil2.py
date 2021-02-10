# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:03:41 2020

@author: stma0003
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as syms
from scipy.stats import norm
from scipy.stats import t
from sympy.core import evalf

""" Berechnung der Toleranz für eine Temperaturmessung mit einem NTC """


""" Parameter aus Aufgabenstellung mit Toleranzangabe übernehmen """

TNOM = 293
PNOM = 1013

Qnom_0 = 0.5
Qnom_min = 0.499
Qnom_max = 0.501
TQnom = Qnom_max - Qnom_min
sigQnom = TQnom/6

pD_0 = 1013
pD_min = 1012
pD_max = 1014
TpD = (pD_max - pD_min)
sigpD = TpD/np.sqrt(12)

TD_0 = 293
TD_min = 292.5
TD_max = 293.5
TTD = (TD_max - TD_min)
sigTD = TTD/np.sqrt(12)

cPE_0 = 50e-6
cPE_min = 48.75e-6
cPE_max = 51.25e-6
TcPE = (cPE_max - cPE_min)
sigcPE = TcPE/6

K_0 = 0.95
K_min = 0.95 - 0.0025
K_max = 0.95 + 0.0025
TK = (K_max - K_min)
sigK = TK/6


""" Masskette definieren und Empfindlichkeiten berechnen """

# Variablen und Funktion symbolisch definieren
Qkor, pD, Tnom, pnom, TD, K, cPE, Qnom = syms.symbols('Qkor, pD, Tnom, pnom, TD, K, cPE, Qnom')
Qkor = pD*Tnom/(pnom*TD*K)*(1+cPE*(pD-pnom))*Qnom

# Empfindlichkeiten symbolisch berechnen
E_Qnom = Qkor.diff(Qnom)
E_pD = Qkor.diff(pD)
E_TD = Qkor.diff(TD)
E_cPE = Qkor.diff(cPE)
E_K = Qkor.diff(K)

# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {pD:pD_0, Tnom:TNOM, pnom:PNOM, TD:TD_0, K:K_0, cPE:cPE_0, Qnom:Qnom_0}
EQnom =  float(E_Qnom.evalf(subs=values))
EpD = float(E_pD.evalf(subs=values))
ETD = float(E_TD.evalf(subs=values))
EcPE = float(E_cPE.evalf(subs=values))
EK = float(E_K.evalf(subs=values))


""" Tolerierung unkorrelierter Groessen """

# Arithmetische Tolerierung
Qkorari = np.abs(EQnom*TQnom) + np.abs(EpD*TpD) + np.abs(ETD*TTD) \
    + np.abs(EcPE*TcPE) + np.abs(EK*TK) 

# Statistische Tolerierung über Grenzwertmethode
Qkorsta = 6*np.sqrt((EQnom*sigQnom)**2 + (EpD*sigpD)**2 + (ETD*sigTD)**2 \
                    + (EcPE*sigcPE)**2 + (EK*sigK)**2)

print(' ')
print('Arithmetische Tolerierung: ', round(Qkorari,3))
print('Statistische Tolerierung über Grenzwertmethode: ', round(Qkorsta,3))


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
print(' ')
print('Statistische Tolerierung über Monte-Carlo-Simulation: ', round(TQkorMC,3))


# """ Statistische Simulation für korrlierte Spannungen """

# # Generieren von korrelierten Zufallszahlen
# z1 = np.random.uniform(-1/2,1/2, N)
# z2 = np.random.uniform(-1/2,1/2, N)
# Uref_kor = TUref*z1
# Uadc_kor= roh*TUadc*z1 + np.sqrt(1-roh**2)*TUadc*z2
# Uref_kor = Uref_0 + Uref_kor
# Uadc_kor = Uadc_0 + Uadc_kor

# # Berechnung der Zielgröße und der statistischen Kennwerte
# N_kor = Rntc_sim/(R_sim + Rntc_sim)*Uref_kor/Uadc_kor + Na_sim
# Nmean = np.mean(N_kor)
# Nstd = np.std(N_kor,ddof=1)
# Fkor = norm.cdf(Nplot,Nmean,Nstd)

# # Toleranz als Prognoseintervall
# TNMC2 = Nstd*np.sqrt(1+1/N)*(c2-c1)
# TTMC2 = np.abs(TNMC2/ET)
# print('Toleranzbereich bei Monte-Carlo-Simulation mit Korrelation: ', round(TTMC2,3))


# """ Grafische Darstellung der Simulation """

# fig = plt.figure(2, figsize=(6, 4))
# fig.suptitle('')
# ax1 = fig.subplots(1,1)
# ax1.plot(Nplot,Fsim,'b',label = 'unkorreliert')
# ax1.plot(Nplot,Fkor,'r:',label = 'korreliert')
# ax1.axis([0.63,0.65,0,1])
# ax1.set_xlabel('ADC-Wert $N$')
# ax1.set_ylabel('Verteilungsfunktion $F(N)$') 
# ax1.grid(True)


# """ Grafische Darstellung der Häufigkeitsverteilungen """

# fig = plt.figure(3, figsize=(12, 4))
# fig.suptitle('')
# ax1, ax2 = fig.subplots(1,2)
# ax1.hist(Uref_kor, 20, density=True, facecolor='b')
# ax1.axis([4.97,5.03,0,30])
# ax1.set_xlabel('Referenzspannung $U_{REF}$ / V')
# ax1.set_ylabel('Relative Häufigkeit $h(U_{REF})$') 
# ax1.grid(True)
# ax2.hist(Uadc_kor, 20, density=True, facecolor='b')
# ax2.axis([4.97,5.03,0,30])
# ax2.set_xlabel('Referenzspannung $U_{ADC}$ / V')
# ax2.set_ylabel('Relative Häufigkeit $h(U_{ADC})$') 
# ax2.grid(True)


# fig = plt.figure(4, figsize=(6, 4))
# fig.suptitle('')
# ax1 = fig.subplots(1,1)
# ax1.plot(Uref_kor, Uadc_kor, 'r+')
# ax1.axis([4.97,5.03,4.97,5.03])
# ax1.set_xlabel('Referenzspannung $U_{REF}$ / V')
# ax1.set_ylabel('ADC-Spannung $U_{ADC}$ / V')
# ax1.grid(True)

# print(' ')
# print('Korrelation der Simulationsdaten: ', round(np.corrcoef(Uref_kor,Uadc_kor)[0,1],3))
