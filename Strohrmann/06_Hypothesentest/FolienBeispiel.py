# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:32:49 2020

@author: stma0003
"""

"""  Initialisierung: Variablen löschen, KOnsole leeren """    
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
    
""" Bibliotheken importieren"""
from scipy.stats import t, norm, chi2, f
import numpy as np

""" 
Hypothesentest zur Zugfestigkeit von Folien
einseitiger Verwerfungsbereich unten,
Varianz ist nicht bekannt, t-Verteilung
"""

""" Werte aus Aufgabe übernehmen"""
alpha = 0.05;
mu0 = 44;
sig0 = 1.29;
Rm = [44.00, 44.50, 44.50, 44.40, 42.50, 40.80,\
      43.30, 43.50, 46.70, 44.90, 43.90, 43.70,\
      42.90, 44.10, 42.00, 44.00, 44.80, 43.50,\
      43.80, 43.80, 42.80, 44.30, 41.10, 45.80,\
      43.20, 44.20, 42.50, 46.10, 44.50, 43.20]

""" Daten auswerten """
N = np.size(Rm)
Rmquer = np.mean(Rm)
s = np.std(Rm,ddof=1)

""" Berechnung des PrognosebereichsAnnahmebereichs für den Mittelwert"""
c1 = t.ppf(alpha,N-1)
Rm_min = mu0 + (c1*sig0/np.sqrt(N))

""" Ausgabe """
print(' ')
print('Untere Annahmegrenze für Rm: ', Rm_min)
print('Stichprobenwert für Rmquer:', Rmquer)

""" Berechnung des Annnahmebereichs für die Varianz 
c2 = chi2.ppf(1-alpha,N-1)
s2_max = c2*sig0**2/(N-1)

 Ausgabe 
print(' ')
print('Obere Annahmegrenze für Varianz: ', s2_max)
print('Stichprobenwert für Varianz:', s**2)
"""