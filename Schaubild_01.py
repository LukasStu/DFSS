# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:22:09 2020

@author: LStue
"""

import numpy as np
import matplotlib.pyplot as plt

# Angabe der bekannten Wahrscheinlichkeiten
PD=100E-6
PEDD=99.99E-2
PEDnotD=0.02E-2

# Berechnung der fehlenden Wahrscheinlichkeiten
PnotD=1-PD

# Defekte Sensoren | Meldung Defekt
PDED=(PEDD*PD)/(PEDD*PD+PEDnotD*PnotD)

# x-Achse in %
x = np.linspace(1E-4,1,10000)
x = x/100 # wegen Prozent

# y-Achse
y = (PEDD*PD)/(PEDD*PD+x*PnotD)

# Ergebnis plotten
fig, ax = plt.subplots()
ax.semilogx(x, y)
ax.plot(PEDnotD,PDED,'o') # Punkt markieren
ax.grid(True, which="both")

#Beschriftung
ax.set_title('Aussagesicherheit in Abhänigkeit der Defektmeldung')
ax.set_xlabel('Wahrscheinlichkeit, mit der ein funktionsfähiger Sensor\n als defekt eingestuft wird $P(ED\mid\overline{D})/\%$')
ax.set_ylabel('Aussagesicherheit $P(D\mid ED)$')



