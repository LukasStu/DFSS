# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:03:41 2020

@author: stma0003
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as syms
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import t
from sympy.core import evalf
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.formula.api import ols

""" Berechnung der Toleranz eines Spannungsteilers """


""" Masskette definieren und Empfindlichkeiten berechnen"""
# Variablen und Funktion symbolisch definieren
Uref, R1, R2, U = syms.symbols('Uref, R1, R2, U')
U = R2/(R1+R2)*Uref

# Empfindlichkeiten symbolisch berechnen
E_Uref = U.diff(Uref)
E_R1 = U.diff(R1)
E_R2 = U.diff(R2)


""" Widerstände mit Toleranzangebane """
R10 = 100
sigR1 = 1
R20 = 100
sigR2 = 1
Uref0 = 5
TUref = 0.05
sigUref = TUref/np.sqrt(12)
U0 = R20/(R10+R20)*Uref0

# Werte definieren und Empfindlichkeiten numerisch berechnen
values = {Uref:Uref0, R1:R10, R2:R20}
EUref =  float(E_Uref.evalf(subs=values))
ER1 = float(E_R1.evalf(subs=values))
ER2 = float(E_R2.evalf(subs=values))


""" Definition der Widerstandsbereiche und ihrer Wahrscheinlichkeitsdichten """
# Definition der gemeinsamen Auflösung im Spannungsbereich
dU = 0.0001;

# Widerstand R1 mit Normalverteilung
UR1min = -0.1
UR1max = 0.1
UR1 = np.arange(UR1min, UR1max+dU, dU)
f1 = norm.pdf(UR1,0,np.abs(ER1*sigR1))

# Widerstand R2 mit Normalverteilung
UR2min = -0.1
UR2max = 0.1
UR2 = np.arange(UR2min, UR2max+dU, dU)
f2 = norm.pdf(UR2,0,np.abs(ER2*sigR2))

# Spannungsquelle mit Gleichverteilung
Urefmin = -0.1
Urefmax = 0.1
dUref = np.arange(Urefmin, Urefmax+dU, dU)
dUrefmin = -0.025
dUrefmax = 0.025
f3 = uniform.pdf(dUref,dUrefmin*EUref,(dUrefmax-dUrefmin)*EUref)


""" Grafische Darstellung der Wahrscheinlichleitsdichten """
fig = plt.figure(1, figsize=(12, 4))
fig.suptitle('')
ax1, ax2, ax3 = fig.subplots(1,3)
ax1.plot(UR1, f1,'b')
ax1.axis([UR1min,UR1max,0,50])
ax1.set_xlabel('Abweichung durch $R_1$ / V')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($U_{R1}$) $\cdot$ V') 
ax1.set_title('Normalverteilung')
ax1.grid(True)
ax2.plot(UR2, f2,'b')
ax2.axis([UR2min,UR2max,0,50])
ax2.set_xlabel('Abweichung durch $R_2$ / V')
ax2.set_ylabel('Wahrscheinlichkeitsdichte f($U_{R2}$) $\cdot$ V') 
ax2.set_title('Normalverteilung')
ax2.grid(True)
ax3.plot(dUref, f3,'b')
ax3.axis([Urefmin,Urefmax,0,50])
ax3.set_xlabel('Abweichung durch $U_{ref}$ / V')
ax3.set_ylabel('Wahrscheinlichkeitsdichte f($U_{ref}$) $\cdot$ V') 
ax3.set_title('Gleichverteilung')
ax3.grid(True)
fig.tight_layout(rect=[0, 0, 0.6, 1])


""" Wahrscheinlichkeitsverteilung des Gesamtwiderstandes über Faltung """
# Faltung der beiden ersten Wahrscheinlichkeitsdchten
f12 = np.convolve(f1,f2)*dU
U12min = UR1min + UR2min
U12max = UR1max + UR2max
U12 = np.arange(U12min,U12max+dU,dU)

# Faltung der dritten Wahrscheinlichkeitsdichte
f123 = np.convolve(f12,f3)*dU
U123min = U12min + Urefmin
U123max = U12max + Urefmax
U123 = np.arange(U123min,U123max+dU,dU)
# Fehlerkorrektur der Länge
U123 = U123[0:np.size(f123)]

# Berechnung Verteiungsfunktion
F123 = np.cumsum(f123)*dU
F123 = F123/np.max(F123)

# Berechnung der Toleranzgrenzen über Ausfallwahrscheinlichkeiten 
indexmin = np.max(np.where(F123 <= (1-0.9973)/2))
indexmax = np.min(np.where(F123 >= (1+0.9973)/2))
UmaxCon = U123[indexmax]
UminCon = U123[indexmin]
TUCon = UmaxCon - UminCon
print(' ')
print('Toleranzbereich bei Faltung: ', round(TUCon,3))


""" Grafische Darstelung des Faltungsergebnisses """
fig = plt.figure(2, figsize=(12, 4))
fig.suptitle('Ergebnisse der Faltungsoperation')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(U123+U0, f123,'b')
ax1.axis([2.35,2.65,0,25])
ax1.set_xlabel('Ausgangsspannung $U$ / V')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($U$) $\cdot$ V') 
ax1.grid(True)
ax2.plot(U123+U0, F123,'b')
ax2.axis([2.35,2.65,0,1])
ax2.set_xlabel('Ausgangsspannung $U$ / V')
ax2.set_ylabel('Verteilungsfunktion F($U$)') 
ax2.grid(True)


""" Vergleich: Rechnung mit Grenzwertmethode """
TUSta = 6*np.sqrt((EUref*sigUref)**2 + (ER1*sigR1)**2 + (ER2*sigR2)**2)
print(' ')
print('Toleranzbereich bei Grenzwertmethode: ', round(TUSta,3))


""" Statistische Simulation und Auswertung als Prognoseintervall """
# Generieren von Zufallszahlen
N = 10000
R1sim = np.random.normal(R10, sigR1, N)
R2sim = np.random.normal(R20, sigR2, N)
Urefsim = np.random.uniform(Uref0-TUref/2, Uref0+TUref/2, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
Usim = R2sim/(R1sim+R2sim)*Urefsim
Umean = np.mean(Usim)
Ustd = np.std(Usim,ddof=1)
Uplot = np.arange(2.35,2.65,0.001);
fsim = norm.pdf(Uplot,Umean,Ustd)
Fsim = norm.cdf(Uplot,Umean,Ustd)

# Toleranz als Prognoseintervall
gamma = 0.9973
c1 = t.ppf((1-gamma)/2,N-1)
c2 = t.ppf((1+gamma)/2,N-1)
TUMC1 = Ustd*np.sqrt(1+1/N)*(c2-c1)
print(' ')
print('Toleranzbereich bei Monte-Carlo-Simulation mit Prognoseintervall: ', round(TUMC1,3))


""" Grafische Darstellung der Simulation """
fig = plt.figure(3, figsize=(12, 4))
fig.suptitle('Ergebnisse der statistischen Simulation')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(Usim,'r+')
ax1.axis([0,N,2.35,2.65])
ax1.set_xlabel('Stichprobe $n$')
ax1.set_ylabel('Ausgangsspannung $U$ / V') 
ax1.grid(True)
ax2.hist(Usim, 10, density=True, facecolor='b')
ax2.plot(Uplot,fsim,'r')
ax2.axis([2.35,2.65,0,25])
ax2.set_xlabel('Ausgangsspannung $U$ / V')
ax2.set_ylabel('Wahrscheinlichkeitsdichte f($U$) $\cdot$ V') 
ax2.grid(True)


""" Statistische Simulation und Auswertung als Häufigkeitsverteilung """
Usort = np.append(np.append(2.35,np.sort(Usim)),np.max(Usim))
Hsort = np.append(np.append(0,np.arange(1,N+1)/N),1)
indexmin = np.max(np.where(Hsort <= (1-0.9973)/2))
indexmax = np.min(np.where(Hsort >= (1+0.9973)/2))
UmaxMC2 = Usort[indexmax]
UminMC2 = Usort[indexmin]
TUMC2 = UmaxMC2 - UminMC2
print('Toleranzbereich bei Monte-Carlo-Simulation mit numerischer Auswertung: ', round(TUMC2,3))


""" Grafischer Vergleich der Häufigkeitsverteilungen """
fig = plt.figure(4, figsize=(12, 4))
fig.suptitle('Grafischer Vergleich der Häufigkeitsverteilungen')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(U123+U0, f123,'b')
ax1.plot(Uplot,fsim,'r')
ax1.axis([2.35,2.65,0,25])
ax1.set_xlabel('Ausgangsspannung $U$ / V')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($U$) $\cdot$ V') 
ax1.grid(True)
ax2.plot(U123, F123,'b',label='Faltung')
ax2.plot(Uplot,Fsim,'r',label='Simulation Prognose')
ax2.plot(Usort,Hsort,'g',label='Simulation Numerisch')
ax2.axis([2.35,2.65,0,1])
ax2.set_xlabel('Ausgangsspannung $U$ / V')
ax2.set_ylabel('Verteilungsfunktion F($U$)') 
ax2.grid(True)
ax2.legend(loc=2)


""" Test der Empfindlichkeiten """
regress = pd.DataFrame({'R1': R1sim,
                        'R2': R2sim,
                        'U': Usim})
model = ols("U ~ R1" , regress).fit()
st, data, ss2 = summary_table(model, alpha=0.05)
regress['Fit R1'] = data[:, 2]
ER1sim = model.params['R1']
model = ols("U ~ R2" , regress).fit()
st, data, ss2 = summary_table(model, alpha=0.05)
regress['Fit R2'] = data[:, 2]
ER2sim = model.params['R2']


""" Grafische Darstellung der Empfindlichkeit """
fig = plt.figure(5, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(regress['R1'],regress['U'],'r+', label='Stichprobe')
ax1.plot(regress['R1'],regress['Fit R1'],'b',label='Regression')
ax1.set_xlabel('Widerstand $R_1$ / $\Omega$')
ax1.set_ylabel('Ausgangsspannung $U$ / V')
ax1.axis([95,105,2.35,2.65])
ax1.grid(True)
ax2.plot(regress['R2'],regress['U'],'r+', label='Stichprobe')
ax2.plot(regress['R2'],regress['Fit R2'],'b',label='Regression')
ax2.set_xlabel('Widerstand $R_2$ / $\Omega$')
ax2.set_ylabel('Ausgangsspannung $U$ / V')
ax2.axis([95,105,2.35,2.65])
ax2.grid(True)
ax2.legend()
print(' ')
print('Empfindlichkeit E_R1 analytisch', round(ER1,3))
print('Empfindlichkeit E_R1 numerisch', round(ER1sim,3))
print('Empfindlichkeit E_R2 analytisch', round(ER2,3))
print('Empfindlichkeit E_R2 numerisch', round(ER2sim,3))



