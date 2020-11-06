# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:48:58 2020

@author: LStue
"""

import numpy as np
import matplotlib.pyplot as plt

u_ref = 5
bit = 8
u_LSB = 5/(2**8)
p_LSB = 1/u_LSB

x = np.linspace(0,u_LSB,10000)
y = p_LSB*np.ones(np.shape(x))

plt.plot(x,y)
plt.xlabel('Spannung/V')
plt.ylabel('Wahrsch.-Dichte/V')
