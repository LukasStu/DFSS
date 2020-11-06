# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:07:02 2020

@author: LStue
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Frästeile
# Infos aus Aufgabe

mu = 2.5
sigma = 0.002

# a) Mit wie viel Prozent Ausschuss muss der Produktionsbetrieb rechnen, wenn alle Teile mindestens
#    2.495 mm stark sein sollen

# Wahrscheinlichkeit für d < 2,495 mm
a = norm.cdf(2.495,mu,sigma)
print('Ausschuss für d<2,495mm/%',a*100)

# b) Mit wie viel Prozent Ausschuss muss der Produktionsbetrieb rechnen, wenn alle Teile höchstens
#    2.506 mm stark sein dürfen.

# Wahrscheinlichkeit für d < 2,506 mm
b = norm.cdf(2.506,mu,sigma)
b = 1 - b 
print('Ausschuss für d>2,506mm/%',b*100)

# c) Mit einem Kunden wurde eine Toleranzgrenze von ± 0.003 mm vereinbart. Mit wie viel Prozent
# Ausschuss ist bei dieser Vereinbarung zu rechnen?

# Wahrscheinlichkeit für 2,497 < d =< 2,503 mm
c = norm.cdf(2.503,mu,sigma) - norm.cdf(2.497,mu,sigma)
c = 1-c
print('Ausschuss für 2,497 < d =< 2,503 mm/%',c*100)

# Der Ausschuss der Produktion soll vermindert werden. Als maximaler Ausschuss soll eine Quote von 2
# % bei einer Toleranzgrenze von ± 0.003 mm zugelassen werden.

# d) Welchen Wert darf die Prozessstreuung maximal haben, um bei gleichbleibendem Erwartungswert
# die Zielvorgaben zu erreichen?

sigma2 = (2.5-2.503)/norm.ppf(0.01,0,1)
print('Sigma',sigma2)

# Durch Alterungserscheinungen am Fräswerkzeug kommt es in der Fertigung zu einer Verschiebung
# des Mittelwertes.
# e) Wie ändert sich die Ausbeute bei einer definierten Toleranzgrenze von ± 0.003 mm, wenn sich
# der Erwartungswert auf 2.51 mm verändert? Gehen Sie von einer Prozessstandardabweichung
# von 0.002 mm aus.

mue = 2.51
sigmae = 0.002
# Wahrscheinlichkeit für 2,497 < d =< 2,503 mm
e = norm.cdf(2.503,mue,sigmae) - norm.cdf(2.497,mue,sigmae)
e = 1-e
print('Ausschuss für 2,497 < d =< 2,503 mm/%',e*100)