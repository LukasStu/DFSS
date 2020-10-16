# -*- coding: utf-8 -*-
"""
Abbildung: Auswertung zum Beispiel Sensordiagnose: 
Wahrscheinlichkeit mit der ein Sensor wirklich defekt ist,
wenn er mit einer Diagnosefunktion als defekt eingestuft wurde
"""

import numpy as np
import matplotlib.pyplot as pl

# Initialisierung

# Datensatz berechnen
x = np.logspace(-4,0,50);
p = np.divide(0.9999*1e-4,(0.9999*1e-4+np.divide(x,100)*(1-1e-4)))

# Ergebnis plotten
fig, ax = pl.subplots(figsize=(6, 4))
ax.semilogx(x,p, 'b', Linewidth = 2)
ax.semilogx(0.02,0.333, 'bo')
ax.grid(True, which="both")
ax.axis([1e-4, 1, 0, 1])
# ax.legend(loc = 'upper right')
# ax.set_title('Beispiel Sensordiagnose', fontsize = 14)
ax.set_xlabel('Wahrscheinlichkeit P(B|$\mathrm{A\')}$ / %')
ax.set_ylabel('Aussagesicherheit P(A|B) / %')
ax.text(0.06,0.4,'P(B|$\mathrm{A\'}$) = 0.02 %\n P(A|B) = 33.3 %', 
        bbox={'facecolor':'white', 'alpha':1, 'pad':5})
pl.show()
