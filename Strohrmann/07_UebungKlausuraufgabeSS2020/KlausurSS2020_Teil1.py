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

""" Hypothesentest zur Bewertung von Düsen """
""" Stichprobenwerte aus Aufgabe übernehmen und als Vektor speichern """

data = loadmat('Durchflussmessung')
X = np.reshape(data['QREF'],-1)

""" Kennwerte berechnen, Histogramm und Boxplot erstellen """

N = np.size(X)
xquer = np.mean(X)
s = np.std(X,ddof=1)
xplot = np.linspace(0.48,0.54,100)
fplot = norm.pdf(xplot,xquer,s)
fig = plt.figure(1, figsize=(12, 4))
ax1, ax2 = fig.subplots(1,2)
ax1.hist(X, 10, density=True, facecolor='b')
ax1.plot(xplot,fplot,'r')
ax1.set_xlabel('Volumenstrom Q / m³/h')
ax1.set_ylabel('Wahrscheinlichkeitsdichte')
ax1.grid(False)
ax1.axis([0.48,0.54,0,80])
ax2.boxplot(X)
ax2.set_ylabel('Volumenstrom Q / m³/h')

""" Berechnung und Ausgabe der Parameter mit Konfidenzbereich """

gamma = 0.95
c1 = t.ppf((1-gamma)/2,N-1)
c2 = t.ppf((1+gamma)/2,N-1)
mu = round(xquer,3)
muc1 = round(xquer - c2*s/np.sqrt(N),3)
muc2 = round(xquer - c1*s/np.sqrt(N),3)
c1 = chi2.ppf((1-gamma)/2,N-1)
c2 = chi2.ppf((1+gamma)/2,N-1)
sig = round(s,3)
sigc1 = round(s*np.sqrt(N/c2),3)
sigc2 = round(s*np.sqrt(N/c1),3)
print(' ')
print('Konfidenzbereiche')
print('Mittelwert : ', muc1, '<=', mu, '<=', muc2)
print('Standardabweichung  : ', sigc1, '<=', sig, '<=', sigc2)

""" Durchführung Hypothesentest """
""" Prüfung Mittelwert auf einen Sollwert mu = 0.5 """
""" unbekannte Varianz, t-Verteilung, beidseitiger Verwerfungsbereich """

mu0 = 0.5
alpha = 0.05
c1 = t.ppf(alpha/2,N-1)
c2 = t.ppf(1-alpha/2,N-1)
Qquer_min = round(mu0 + c1*s/np.sqrt(N),3)
Qquer_max = round(mu0 + c2*s/np.sqrt(N),3)
print(' ')
print('Hypothesentest')
print('Untere Annahmegrenze für keine Abweichung / m³/h: ', Qquer_min)
print('Obere Annahmegrenze für keine Abweichung / m³/h ', Qquer_max)
print('Mittelwert des Stichprobe liegt außerhalb des Annahmebereichs')
print('Prüfstand hat eine signifikante Abweichung')

""" Berechnung und Darstellung der Gütefunktion """

dQ = np.linspace(-0.02,0.02,1000)
Guete = 1 + t.cdf((Qquer_min - (dQ+mu0))/s*np.sqrt(N),N-1) - t.cdf((Qquer_max - (dQ+mu0))/s*np.sqrt(N),N-1)
fig = plt.figure(2, figsize=(6, 4))
ax3 = fig.subplots(1,1)
ax3.plot(dQ,Guete,'b')
ax3.set_xlabel('Abweichung / m³/h')
ax3.set_ylabel('Gütefunktion')
ax3.grid(True)
ax3.axis([-0.02,0.02,0,1])

""" Wahrscheinlichkeit der Erkennung einer Sensordrift von - 5 %rF """

index = np.min(np.where(Guete <= 0.95))
Q95 = round(dQ[index],4)
print(' ')
print('Gütefunktion')
print('Abweichung mit einer Erkennung von 95 %  / m³/h: ', Q95)
