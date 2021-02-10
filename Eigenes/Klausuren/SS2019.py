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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
from statsmodels.formula.api import ols

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('Signifikanz')

# n = 1000 1/min
m1 = np.array(data['m1']).reshape(-1)
m1_quer = np.mean(m1)
s_m1 = np.std(m1 ,ddof=1)
N = m1.size

# n = 2140 1/min
m2 = np.array(data['m2']).reshape(-1)
m2_quer = np.mean(m2)
s_m2 = np.std(m2 ,ddof=1)
M = m2.size

"""a) Ist die Drehzahl n signifikant für das Schleppmoment? Führen Sie mit den
angegebenen Daten einen Hypothesentest auf gleichen Mittelwert durch.
Wie lautet die Nullhypothese, wie die Alternative?
Welche Zufallsvariable wählen Sie für den Hypothesentest?"""

# Hypothesentest auf gleiche Mittelwerte
# H0: Beide Stichproben stammen aus einer Grundgesamtheit und haben gleiche Mittelwerte
# H1: Beide Stichproben stammen nicht aus einer Grundgesamtheit und haben unterschiedliche Mittelwerte
# t-Zufallsvariable: t = (m1_quer-m2_quer)/(np.sqrt(1/N+1/M)*s)
# s = np.sqrt((N-1)*s_m1+(M-1)*s_m2/(N+M-2))

gamma = 0.95
s = np.sqrt(((N-1)*s_m1+(M-1)*s_m2)/(N+M-2))
c_min = t.ppf((1-gamma)/2, N+M-2)
c_max = t.ppf((1+gamma)/2, N+M-2)
delta_m_Annahme = np.array([c_min*np.sqrt(1/N+1/M)*s, c_max*np.sqrt(1/N+1/M)*s])

# Vergleich mit Stichprobe
if (m1_quer-m2_quer)<delta_m_Annahme[0] or (m1_quer-m2_quer)>=delta_m_Annahme[1]:
    print("Hypothese verworfen")
else:
    print("Hypothese angenommen")
    print("{:.4f} < {:.4f} <= {:.4f}".format(delta_m_Annahme[0],(m1_quer-m2_quer),delta_m_Annahme[1]))
 
    
 
    
"""b) Erstellen Sie für den Hypothesentest die zugehörige Gütefunktion.
Welche Abweichung können Sie mit einer Wahrscheinlichkeit von 99 % erkennen?"""
# Gütefunktion
d_mu = np.linspace(-1, 1, num=10000)
G = t.cdf((delta_m_Annahme[0]+d_mu)/(np.sqrt(1/N+1/M)*s), N+M-2) + 1-t.cdf((delta_m_Annahme[1]+d_mu)/(np.sqrt(1/N+1/M)*s), N+M-2)
# Berechnung der Abweichung für Erkennung 99%-Erkennung
index = np.min(np.where(G <= 0.99))
print("Mit 99%-tiger Wahrscheinlichkeit wird eine Abweichung von {:.4f}Nm erkannt".format(d_mu[index]))

# Kontrollplot
fig, ax = plt.subplots()
ax.plot(d_mu,G,label=r'$N=%d$'%N)
ax.set_xlabel(r'$\Delta\,\overline{M}$/Nm')
ax.set_ylabel(r'$1-\beta(\mu_1)$')
ax.plot(d_mu[index],G[index],'r+')
ax.grid(True)


"""c) Plausibilisieren Sie das Ergebnis des Hypothesentests
mit einer geeigneten grafischen Darstellung."""
df_box = pd.DataFrame({'m1': m1,
                       'm2': m2})
fig2, ax2 = plt.subplots()
df_box.boxplot(column=['m1', 'm2'])





"""d) Können Sie mit Hilfe einer Korrelationsanalyse bewerten, ob die Drehzahl n einen
    signifikanten Einfluss aufweist? Führen Sie auch dazu einen geeigneten
    Hypothesentest durch. Wie lautet die verwendete Zufallsvariable?"""

df = pd.DataFrame({'n': np.repeat([1000, 2140], 8),
                   'M': np.concatenate((m1,m2))})

""" Berechnung Korrelation und Hypothesentest roh = 0 über scipy.stats """
Corr, pval = stats.pearsonr(df['n'],df['M'])

if pval >=0.05:
    print("Keine Korrelation, H0 gilt")
else:
    print("Korrelation vorhanden, H1 gilt")

