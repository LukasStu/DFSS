# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:41:38 2020

@author: LStue
"""
#Schmelzwärme

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


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
VA = loadmat('Schmelzwaerme')['VA']
VA = np.array(VA).reshape(-1)

VB = loadmat('Schmelzwaerme')['VB']
VB = np.array(VB).reshape(-1)

"a) Mittelwert von VA + 95% Konfidenzintervall"
VA_quer = np.mean(VA)
s_VA = np.std(VA,ddof=1)
N_VA = VA.size

# Konfidenzzahlen
gamma95 = 0.95

# Grenzen für 95, t-verteilt%
c1_95_mu = t.ppf((1-gamma95)/2,N_VA-1)
c2_95_mu = t.ppf((1+gamma95)/2,N_VA-1)

# 95% Konfidenzintervall
mu_VA_min_95 = VA_quer-c2_95_mu*s_VA/np.sqrt(N_VA)
mu_VA_max_95 = VA_quer-c1_95_mu*s_VA/np.sqrt(N_VA)

print('a) Konfidenzintervall mu_VA:\n{:.4f} < {:.4f} =< {:.4f}\n'.format(mu_VA_min_95,VA_quer,mu_VA_max_95))

"b) Zweiseitges Konfidenzintervall für sigma"
# Grenzen für 95%
c1_95_sig = chi2.ppf((1-gamma95)/2,N_VA-1)
c2_95_sig = chi2.ppf((1+gamma95)/2,N_VA-1)

# 95% Konfidenzintervall
sigma_min_95 = np.sqrt((N_VA-1)/c2_95_sig)*s_VA
sigma_max_95 = np.sqrt((N_VA-1)/c1_95_sig)*s_VA

print('b) Konfidenzintervall sig²_VA:\n{:.6f} < {:.6f} =< {:.6f}\n'.format(sigma_min_95**2,s_VA**2,sigma_max_95**2))

"c) Histogramm"
"b) Histogramm 1"
# Wahrscheinlichkeitsverteilung
fig, ax = plt.subplots()
ax.hist(VA, density='true',bins=round(np.sqrt(N_VA)),label='Wahrscheinlichkeitsverteilung VA')
# Grundgesamtheit
xaxes = np.linspace(np.min(VA)*0.9995,np.max(VA)*1.001,1000)
pdf = norm.pdf(xaxes,loc=VA_quer,scale=s_VA)
ax.plot(xaxes,pdf,'r',label='Wahrscheinlichkeitsdichte der Ggh')
ax.set_xlabel(r'Wärme in $\frac{\mathrm{cal}}{\mathrm{g}}$')
ax.set_ylabel(r'Wahrscheinlichkeit')
ax.legend()

"d) Hypothesentest auf identische Mittelwerte mu_VA und mu_VA, "
# Stichprobenanalyse
VB_quer = np.mean(VB)
s_VB = np.std(VB,ddof=1)
N_VB = VB.size

s_gesamt = np.sqrt(((N_VA-1)*s_VA+(N_VB-1)*s_VB)/N_VA+N_VB-2)
# Intervallgrenzen der t-verteilen ZV mit NA+NB-2 Freiheitsgraden
C = t.interval(gamma95,N_VA+N_VB-2)
# Annahmebereich berechnen
Annnahme_delta_x_quer = np.array([C[0]*np.sqrt(1/N_VA+1/N_VB)*s_gesamt,C[1]*np.sqrt(1/N_VA+1/N_VB)*s_gesamt])

# Vergleich mit Stichprobe
if (VA_quer-VB_quer)<Annnahme_delta_x_quer[0] or (VA_quer-VB_quer)>=Annnahme_delta_x_quer[1]:
    print("Hypothese verworfen")
else:
    print("Hypothese angenommen")
    print("{:.4f} < {:.4f} <= {:.4f}".format(Annnahme_delta_x_quer[0],(VA_quer-VB_quer),Annnahme_delta_x_quer[1]))

# Alternativ p-Value
P = t.cdf((VA_quer-VB_quer)/(np.sqrt(1/N_VA+1/N_VB)*s_gesamt),N_VA+N_VB-2)
    

    
    
    