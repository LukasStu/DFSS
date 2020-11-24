# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:11:45 2020

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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien

a = np.arange(10)
print(a)
print(np.where(a < 5, a, 10*a))
# Wenn a < 5 nehme a, ansonsten 10*a
