# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:21:59 2020

@author: LStue
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t

x = np.linspace(-5,5,10000)
std = norm.pdf(x,0,1)
t5 = t.pdf(x,5)
t50 = t.pdf(x,50)

# Ergebnis plotten
fig, ax = plt.subplots()
ax.plot(x,std,label='Standardnormalverteilung')
ax.plot(x,t5,label='t-Verteilung mit $N=5$')
ax.plot(x,t50,label='t-Verteilung mit $N=50$')
ax.set_title('Vergleich zwischen Standardnormalverteilunf und t-Verteilung')
ax.grid()
ax.legend()