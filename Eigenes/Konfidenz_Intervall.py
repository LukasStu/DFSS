# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:17:08 2020

@author: lukas
Konfindenzintervall von Linearreglern bei
1) Bekannter Varianz (der Grundgesamtheit)
2) Unbekannter Varianz (Schätzung über Stichprobenvarianz)
"""

from scipy import stats
import numpy as np

# Informationen über die Stichprobe
N = 100
U_quer = 5.0
sigma = np.sqrt(0.09)

# Konfidenzzahl
gamma = 0.95

# 1)
# Hier Varianz der Grundgesamtheit bekannt, Varianz des Stichprobenmittelwertes eingesetzt 
ci1 = stats.norm.interval(gamma, loc=U_quer, scale=sigma/np.sqrt(N))

# 2)
# Hier Varianz geschätzt (=Stichprobenvarianz), daraus Varianz des Stichprobenmittelwert
s = np.sqrt(0.09)
ci2 = stats.t.interval(gamma, N, loc=U_quer, scale=s/np.sqrt(N))