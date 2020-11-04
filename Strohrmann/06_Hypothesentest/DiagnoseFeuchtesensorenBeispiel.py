# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""


""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import t, norm, chi2, f

""" Hypothesentest zur Diagnose von Feuchtesensoren """
""" beidseitiger Verwerfungsbereich """
""" Varianz ist bekannt, Standardnormal-Verteilung"""

""" Stichprobenwerte aus Aufgabe übernehmen """
""" Vereinigungsmenge aller Daten ist Grundlage für die Aufgabe """

data = loadmat('ZusammenfassungErprobung85')
X = np.concatenate((data['HighTemp'],data['LowTemp'],
                    data['MesswerteErgaenzung'],data['Moisture'],
                    data['RapidTChange'],data['TempCycle']),axis=1)
X = np.reshape(X,-1)

""" Kennwerte berechnen, Histogramm und Boxplot erstellen """

N = np.size(X)
xquer = np.mean(X)
s = np.std(X,ddof=1)
xplot = np.linspace(-2,2,100)
fplot = norm.pdf(xplot,xquer,s)
fig = plt.figure(1, figsize=(12, 4))
ax1, ax2 = fig.subplots(1,2)
ax1.hist(X, 20, density=True, facecolor='b')
ax1.plot(xplot,fplot,'r')
ax1.set_xlabel('Abweichung / %rF')
ax1.set_ylabel('Wahrschenlichkeitsdichte')
ax1.grid(False)
ax1.axis([-2,2,0,0.8])
ax2.boxplot(X)
ax2.set_ylabel('Abweichung / %rF')

""" Berechnung und Ausgabe des Annahmebereichs für den Mittelwert"""

alpha = 10e-6
c1 = norm.ppf(alpha/2)
c2 = norm.ppf(1-alpha/2)
rF_min = round(c1*s*np.sqrt(2),3)
rF_max = round(c2*s*np.sqrt(2),3)
print(' ')
print('Untere Annahmegrenze für gleiche Ergebnisse / rF: ', rF_min)
print('Obere Annahmegrenze für gleiche Ergebnisse / rF: ', rF_max)

""" Berechnung und Darstellung der Gütefunktion """

dmu = np.linspace(-10,10,10000)
Guete = 1 + norm.cdf((rF_min - dmu)/np.sqrt(2)/s) - norm.cdf((rF_max - dmu)/np.sqrt(2)/s)
fig = plt.figure(2, figsize=(6, 4))
ax3 = fig.subplots(1,1)
ax3.plot(dmu,Guete,'b')
ax3.set_xlabel('Abweichung / %rF')
ax3.set_ylabel('Gütefunktion')
ax3.grid(True)
ax3.axis([-10,10,0,1])

""" Wahrscheinlichkeit der Erkennung einer Sensordrift von - 5 %rF """

index = np.min(np.where(dmu >= - 5))
P = round(Guete[index],3)
print(' ')
print('Wahrscheinlichkeit der Erkennung einer Sensordrift von - 5 %rF: ', P)
