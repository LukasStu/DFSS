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



"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('<Dateiname>')['data']
X = np.array(data).reshape(-1)

x_quer = np.mean(x)
s = np.std(x,ddof=1)
N = x.size

print(' ')
print('Mittelwert x: ', x_quer)
print('Standardabweichung s: ', s)


"""https://matplotlib.org/3.1.1/tutorials/text/mathtext.html"""
" Histogramme"
test=np.array([1,2,2,3,3,3,4,4,4,4])
#absolute Häufigkeit
fig1, ax1 = plt.subplots()
ax1.hist(test)
ax.set_xlabel(r'')
ax.set_ylabel(r'Absolute Häufigkeit')
# relative Häufigkeit
fig2, ax2 = plt.subplots()
ax2.hist(test, weights=np.zeros_like(test) + 1. / test.size)
ax.set_xlabel(r'')
ax.set_ylabel(r'Relative Häufigkeit')
# Wahrscheinlichkeitsverteilung
fig3, ax3 = plt.subplots()
ax3.hist(test, density='true')
ax.set_xlabel(r'')
ax.set_ylabel(r'Wahrscheinlichkeit')

"""Hypothesentest Univariat"""
"""Varianz bekannt -> ZV ist Standardnormalverteilt, wenn H0 gilt"""
#mu1 > mu0
c_max = norm.ppf(1-alpha)
x_quer_max = mu0+(c_max*sigma)/np.sqrt(N)
print("Annahme H0 bei x_quer < {:.4f}".format(x_quer_max)
# Gütefunktion für mu_1 > mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1-norm.cdf((x_quer_min-(mu_0+d_mu))/(sigma/np.sqrt(N)))

#mu1 < mu0
c_min = norm.ppf(alpha)
x_quer_max = mu0+(c_max*sigma)/np.sqrt(N)
print("Annahme bei x_quer > {:.4f}".format(x_quer_min)
# Gütefunktion für mu_1 < mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = norm.cdf((x_quer_max-(mu_0+d_mu))/(sigma/np.sqrt(N)))

      
#mu1 != mu0
c_min = norm.ppf(alpha/2)
c_max = norm.ppf(1-alpha/2)
x_quer_Annahme = np.array([mu_0+c_min*sigma/np.sqrt(N),mu_0+c_max*sigma/np.sqrt(N)])
print("Annahme bei {:.4f} < x_quer <= {:.4f}".format(x_quer_Annahme[0],x_quer_Annahme[1])
# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1+norm.cdf((x_quer_Annahme[0]-(mu_0+d_mu))/(sigma/np.sqrt(N)))-norm.cdf((x_quer_Annahme[1]-(mu_0+d_mu))/(sigma/np.sqrt(N)))

fig, ax = plt.subplots()
ax.plot(d_mu,G,label='Gütefunktion')
ax.set_xlabel(r'$\Delta \mu$')
ax.set_ylabel(r'$1-\beta(\mu_1)$')
ax.grid(True)

""" Korrelationsanalyse """
""" H_0: s_alpha² = 0 """
""" H_0 = Es gibt keinen signifikaten Einfluss zwischen den Gruppen"""