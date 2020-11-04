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
Toleranzaussage ergibt sich aus Prognosebereich
Mittelwert der Grundgesamtheit ist bekannt mu,
Standardabweichung muss geschätzt werden,
deshalb t-Verteilung verwenden 
"""

""" Werte aus Aufgabe übernehmen"""
gamma = 0.95;
D = [49.99, 46.45, 47.50, 50.00, 49.48, 
     49.59, 51.49, 48.94, 49.63, 48.59, 
     48.24, 50.42, 50.33, 51.91, 50.59]

""" Daten auswerten """
N = np.size(D)
mu = 49.5
s = np.std(D,ddof=1)

""" Berechnung des Prognosebereichs """
c1 = t.ppf((1-gamma)/2,N-1)
c2 = t.ppf((1+gamma)/2,N-1)
D_min = mu + (c1*s)
D_max = mu + (c2*s)

""" Ausgabe """
print(' ')
print('Untere Grenze für D: ', D_min)
print('Obere Grenze für D:', D_max)

