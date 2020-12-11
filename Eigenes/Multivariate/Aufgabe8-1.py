
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:00:00 2020

@author: Lukas Stürmlinger Matrikel-Nummer:
"""
""" Aufgabe 8.1: Reihenschaltung von Widerständen """
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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien

""" a) P_0 berechnen """

delta_R1 = 1050 - 950
delta_R2 = 2600 - 2400
P_0 = 1/(delta_R1*delta_R2)
print("P_0 = {} \n".format(P_0))

""" b) Wahrscheinlichkeitsdicht """
# Erzeugung der Widerstandsvektoren
R1 = np.linspace(950,1050, num = 1000)
R2 = np.linspace(2400,2600, num = 1000)

# Erzeugung der Matrizen für die Grafiken
R1,R2 = np.meshgrid(R1,R2)
P0 = P_0*(R1>950)*(R1<=1050)*(R2>2400)*(R2<=2600)

# Definition der Grafik
fig1 = plt.figure()
ax = Axes3D(fig1)
surf = ax.plot_surface(R1,R2,P0*1e6,cmap=cm.Blues_r)
ax.set_xlabel('Widerstand $R_1 / \Omega$')
ax.set_ylabel('Widerstand $R_2 / \Omega$')
ax.set_zlabel('F($R_1,R_2$)')
ax.set_title('Wahrscheinlichkeitsdichte')

""" Ab hier nervts... erst einmal nach hinten verschoben...