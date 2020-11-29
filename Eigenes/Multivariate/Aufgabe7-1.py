# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:01:37 2020

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

data = loadmat('Glasuntersuchung')['values']
#n = np.array(data[values])