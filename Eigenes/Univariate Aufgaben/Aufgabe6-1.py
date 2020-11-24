# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:35:50 2020

@author: LStue
"""
# Statistische Prozesskontrolle

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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien

mu_0 = 3
sigma = 0.5
alpha = 5/100
N = 5

# Berechnung der Intervallgrenzen von z
c1 = norm.ppf(alpha/2)
c2 = norm.ppf(1-alpha/2)

# Berechnung der Eingriffsgrenzen für x_quer
x_quer_Annahme = np.array([mu_0+c1*sigma/np.sqrt(N),mu_0+c2*sigma/np.sqrt(N)])
print("{:.4f} < x_quer <= {:.4f}".format(x_quer_Annahme[0],x_quer_Annahme[1]))

# Gütefunktion für mu_1 != mu_0
d_mu = np.linspace(-2, 2, num=10000)
G = 1+norm.cdf((x_quer_Annahme[0]-(mu_0+d_mu))/(sigma/np.sqrt(N)))-norm.cdf((x_quer_Annahme[1]-(mu_0+d_mu))/(sigma/np.sqrt(N)))

fig, ax = plt.subplots()
ax.plot(d_mu,G,label=r'$N=%d$'%N)
ax.set_xlabel(r'$\Delta \mu$')
ax.set_ylabel(r'$1-\beta(\mu_1)$')
ax.grid(True)


# Berechnung der Wahrscheinlichkeit für Erkennung von delta_mu = 0.5
index = np.min(np.where(d_mu >= 0.5))
print("Wahrscheinlichkeit für die Erkennung einer Abweichung von 0.5 ist {:.4%}".format(G[index]))

# Punkt markieren
ax.plot(d_mu[index],G[index],'r+')

# Notwendiger Stichprobenumfang für 1-beta(0,5) = 95%
# N solange erhöhen, bis Wahrscheinlichkeit>=95%
#fig2, ax2 = plt.subplots()
P=0
n=0
while P<0.95:
    n+=1
    # Neue Eingriffsgrenzen berechnen
    x_quer_Annahme2 = np.array([mu_0+c1*sigma/np.sqrt(n),mu_0+c2*sigma/np.sqrt(n)])
    # Neue Gütefunktion berechnen
    G2 = 1+norm.cdf((x_quer_Annahme2[0]-(mu_0+d_mu))/(sigma/np.sqrt(n)))-norm.cdf((x_quer_Annahme2[1]-(mu_0+d_mu))/(sigma/np.sqrt(n)))
    index2 = np.min(np.where(d_mu >= 0.5))
    P = G2[index2]
    print("Wahrscheinlichkeit für die Erkennung einer Abweichung von 0.5 bei {} Stichproben ist {:.4%}".format(n,P))
    #ax2.plot(d_mu,G2,label=r'$N=%d$'%n)

#ax2.legend()
# Neue Gütefunktion plotten    
ax.plot(d_mu,G2,label=r'$N=%d$'%n)
ax.plot(d_mu[index2],G2[index2],'r+')
ax.legend()    