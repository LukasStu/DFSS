# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:36:04 2020

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
from scipy.stats import t, norm, chi2, f
from scipy.io import loadmat # Für mat-Dateien
import statsmodels.api as sm
from statsmodels.formula.api import ols

Q_norm = 0.5 #m³/h
TQ_norm = 0.002 #m³/h
sig_Q_norm = TQ_norm/6

p_D = 1013 #mbar
Tp_D = 2 #mbar

T_D = 293 #K
TT_D = 1 #K

c_PE = 5E-5 #1/mbar
t_c_PE = 2.5E-6 #1/mbar
sig_c_PE = t_c_PE/6

K = 0.95
TK = 0.005
sig_K = TK/6

"""a) Grenzwertmethode"""
# Umrechnung der Toleranzangabe in Standardabweichung
sig_p_D = Tp_D/np.sqrt(12)
