# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 06:35:08 2020

@author: stma0003
"""

from scipy.stats import norm

""" Variablendefinition """
mu = 998
sigma = 2

""" Berechnung des Ausschussanteils von Kunde A und B ohne Zentrierung """
A_A = norm.cdf(990,mu,sigma) + 1 - norm.cdf((1010-mu)/sigma)
A_B = norm.cdf((980-mu)/sigma) + 1 - norm.cdf((1020-mu)/sigma)
print(' ')
print('Ausschussanteil Kunde A: ', A_A)
print('Ausschussanteil Kunde B: ', A_B)

"""  Berechnung des Ausschussanteils von Kunde A und B nach Zentrierung """
mu_neu = 1000;
A_A = norm.cdf((990-mu_neu)/sigma) + 1 - norm.cdf((1010-mu_neu)/sigma)
A_B = norm.cdf((980-mu_neu)/sigma) + 1 - norm.cdf((1020-mu_neu)/sigma)
print(' ')
print('Ausschussanteil Kunde A nach Zentrierung: ', A_A)
print('Ausschussanteil Kunde B nach Zentrierung: ', A_B)

