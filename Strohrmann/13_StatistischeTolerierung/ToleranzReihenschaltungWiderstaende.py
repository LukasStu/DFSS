# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:03:41 2020

@author: stma0003
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform

""" Berechnung der Toleranz einer Reihenschaltung von Widerständen """


""" Widerstände mit Toleranzangabe """
R10 = 1000
TR1 = 20 
sigR1 = TR1/6
R20 = 500
TR2 = 10
R30 = 800
TR3 = 16


""" Arithmetische Tolerierung bei symmetrischem Toleranzbereich """
RminAri = R10 - TR1/2 + R20 - TR2/2 + R30 - TR3/2
RmaxAri = R10 + TR1/2 + R20 + TR2/2 + R30 + TR3/2
TRAri = RmaxAri - RminAri
print(' ')
print('Toleranzbereich bei arithmetischer Tolerierung: ', round(TRAri,3))

""" Definition der Widerstandsbereiche und ihrer Wahrscheinlichkeitsdichten """
# Definition der gemeinsamen Auflösung
dR = 0.01;

# Widerstand R1 mit Normalverteilung
R1min = 950
R1max = 1050
R1 = np.arange(R1min, R1max+dR, dR)
f1 = norm.pdf(R1,R10,sigR1)

# Widerstand R2 mit Gleichverteilung
R2min = 450
R2max = 550
R2 = np.arange(R2min, R2max+dR, dR)
f2 = uniform.pdf(R2,R20-TR2/2,TR2)

# Widerstand R3 mit Dreieckverteilung
R3min = 750
R3max = 850
R3 = np.arange(R3min, R3max+dR, dR)
R3A = R30 - TR3/2 
R3B = R30 + TR3/2
f3 = (4/(R3B-R3A)**2*(R3-R3A)*(R3<R30) + 4/(R3B-R3A)**2*(R3B-R3)\
      *(R3>=R30))*(R3 >= R3A)*(R3<=R3B)
   

""" Grafische Darstellung der Wahrscheinlichleitsdichten """
fig = plt.figure(1, figsize=(12, 4))
fig.suptitle('')
ax1, ax2, ax3 = fig.subplots(1,3)
ax1.plot(R1, f1,'b')
ax1.axis([R1min,R1max,0,0.15])
ax1.set_xlabel('Widerstand $R_1$ / $\Omega$')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($R_1$) $\cdot$ $\Omega$') 
ax1.set_title('Normalverteilung')
ax1.grid(True)
ax2.plot(R2, f2,'b')
ax2.axis([R2min,R2max,0,0.15])
ax2.set_xlabel('Widerstand $R_2$ / $\Omega$')
ax2.set_ylabel('Wahrscheinlichkeitsdichte f($R_2$) $\cdot$ $\Omega$') 
ax2.set_title('Gleichverteilung')
ax2.grid(True)
ax3.plot(R3, f3,'b')
ax3.axis([R3min,R3max,0,0.15])
ax3.set_xlabel('Widerstand $R_3$ / $\Omega$')
ax3.set_ylabel('Wahrscheinlichkeitsdichte f($R_3$) $\cdot$ $\Omega$') 
ax3.set_title('Dreieckverteilung')
ax3.grid(True)
fig.tight_layout(rect=[0, 0, 0.6, 1])

""" Wahrscheinlichkeitsverteilung des Gesamtwiderstandes über Faltung """
# Faltung der beiden ersten Wahrscheinlichkeitsdchten
f12 = np.convolve(f1,f2)*dR
R12min = R1min + R2min
R12max = R1max + R2max
R12 = np.arange(R12min,R12max+dR,dR)

# Faltung der dritten Wahrscheinlichkeitsdichte
f123 = np.convolve(f12,f3)*dR
R123min = R12min + R3min
R123max = R12max + R3max
R123 = np.arange(R123min,R123max+dR,dR)
# Fehlerkorrektur der Länge
R123 = R123[0:np.size(f123)]

# Berechnung Verteiungsfunktion
F123 = np.cumsum(f123)*dR
F123 = F123/np.max(F123)

# Berechnung der Toleranzgrenzen über Ausfallwahrscheinlichkeiten 
indexmin = np.min(np.where(F123 >= (1-0.9973)/2))
indexmax = np.max(np.where(F123 <= (1+0.9973)/2))
RmaxCon = R123[indexmax]
RminCon = R123[indexmin]
TRCon = RmaxCon -RminCon
print(' ')
print('Toleranzbereich bei Faltung: ', round(TRCon,3))


""" Grafische Darstellung der Wahrscheinlichleitsdichten """
fig = plt.figure(2, figsize=(12, 4))
fig.suptitle('')
ax1, ax2 = fig.subplots(1,2)
ax1.plot(R123, f123,'b')
ax1.axis([2260,2340,0,0.1])
ax1.set_xlabel('Widerstand $R$ / $\Omega$')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($R$) $\cdot$ $\Omega$') 
ax1.grid(True)
ax2.plot(R123, F123,'b')
ax2.axis([2260,2340,0,1])
ax2.set_xlabel('Widerstand $R$ / $\Omega$')
ax2.set_ylabel('Verteilungsfunktion F($R$)') 
ax2.grid(True)


""" Statistische Tolerierung mit Grenzwertsatz """
# Umrechnung Toleranzangabe in Standardabweichung 
sigR2Indi = TR2/np.sqrt(12)
sigR3Indi = TR3/np.sqrt(24)

# Anwendung Grenzwertmethode
RminSta = R10 + R20 + R30 - 3*np.sqrt(sigR1**2+sigR2Indi**2+sigR3Indi**2)
RmaxSta = R10 + R20 + R30 + 3*np.sqrt(sigR1**2+sigR2Indi**2+sigR3Indi**2)
TRSta = RmaxSta - RminSta

# Berechnung der zugrundeliegenden Wahrscheinlicheitsdichte
fSta = norm.pdf(R123,R10 + R20 + R30, \
                  np.sqrt(sigR1**2+sigR2Indi**2+sigR3Indi**2))
print(' ')
print('Toleranzbereich über Grenzwertmethode: ', round(TRSta,3))


""" Grafischer Vergleich Faltung und Grenzwertsatz """
fig = plt.figure(3, figsize=(6, 4))
fig.suptitle('')
ax1 = fig.subplots(1,1)
ax1.plot(R123, f123,'b',label='Faltung')
ax1.plot(R123, fSta, 'r',label='Grenzwertsatz')
ax1.axis([2260,2340,0,0.1])
ax1.set_xlabel('Widerstand $R$ / $\Omega$')
ax1.set_ylabel('Wahrscheinlichkeitsdichte f($R$) $\cdot$ $\Omega$') 
ax1.grid(True)
ax1.legend()


""" Grafische Vergleich der berechneten Toleranzen """
fig = plt.figure(4, figsize=(6, 4))
fig.suptitle('')
ax1 = fig.subplots(1,1)
ax1.plot([1, 1], [RmaxAri, RminAri],'ro-',label='Arithmetisch')
ax1.plot([2, 2], [RmaxCon, RminCon],'go-',label='Faltung')
ax1.plot([3, 3], [RmaxSta, RminSta],'bo-',label='Grenzwertmethode')
ax1.axis([0, 4,2270,2330])
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(('Arithmetisch', 'Faltung', 'Grenzwertmethode'))
ax1.set_ylabel('Toleranzbereich $T_R$ / $\Omega$') 
ax1.grid(True)

