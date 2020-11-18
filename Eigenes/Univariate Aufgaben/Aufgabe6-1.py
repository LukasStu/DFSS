# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:35:50 2020

@author: LStue
"""

"""  Initialisierung: Variablen löschen, KOnsole leeren """    
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien

mu_0 = 3
sigma = 0.5
alpha = 5/100
N = 5

# Berechnung der Intervallgrenzen von z
c1 = stats.norm.ppf(alpha/2)
c2 = stats.norm.ppf(1-alpha/2)

# Berechnung der Eingriffsgrenzen für x_quer
x_quer_Annahme = np.array([mu_0+c1*sigma/np.sqrt(N),mu_0+c2*sigma/np.sqrt(N)])
print("{:.4f} < x_quer <= {:.4f}".format(x_quer_Annahme[0],x_quer_Annahme[1]))

# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1+stats.norm.cdf((x_quer_Annahme[0]-(mu_0+d_mu))/(sigma/np.sqrt(N)))-stats.norm.cdf((x_quer_Annahme[1]-(mu_0+d_mu))/(sigma/np.sqrt(N)))

fig, ax = plt.subplots()
ax.plot(d_mu,G,label='Gütefunktion')
ax.set_xlabel(r'$\Delta \mu$')
ax.set_ylabel(r'$1-\beta(\mu_1)$')
ax.grid(True)