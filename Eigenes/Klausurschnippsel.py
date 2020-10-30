# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:13:09 2020

@author: LStue
"""

""" Bibliotheken importieren"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat # Für mat-Dateien


"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('<Dateiname>')['data']
X = np.array(data).reshape(-1)