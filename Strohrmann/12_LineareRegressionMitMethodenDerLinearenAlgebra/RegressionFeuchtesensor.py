# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:03:41 2020

@author: stma0003
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

""" Messdaten laden """

data = loadmat('Feuchtesensor')
regress = pd.DataFrame({'Cp' : np.reshape(data['Cp'],-1),
                        'rF' : np.reshape(data['RF'],-1),
                        'T' : np.reshape(data['T'],-1)})

""" Messdaten auf unterschiedliche Art grafisch darstellen """

fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1,1)
ax1.plot(regress['T'],regress['rF'],'bo')
ax1.axis([0,100,0,100])
ax1.set_xlabel('Temperatur $T$ / °C');
ax1.set_ylabel('Relative Feuchte $rF$ / %');  
ax1.set_title('Visualisierung Versuchsraums');  
ax1.grid(True)

fig2 = plt.figure(2, figsize=(12, 8))
ax1 = fig2.subplots(1,1)
pd.plotting.scatter_matrix(regress, alpha=1, ax = ax1, Color='b', hist_kwds=dict(Color='b'))
fig2.suptitle('Matrix von Scatterplots')

fig3 = plt.figure(3, figsize=(6, 4))
fig3.suptitle('')
ax1 = fig3.subplots(1,1)
ax1 = Axes3D(fig3)
ax1.scatter(regress['T'], regress['rF'], regress['Cp'],Color='b', alpha=1)
plt.title('Visualisierung der Messpunkte')
ax1.set_xlabel('$T$ / °C')
ax1.set_ylabel('$rF$ / %')
ax1.set_zlabel('$C$ / pF')

""" Regression als Matrizengleichung """

Y = regress['Cp'].to_numpy(copy=True)
Xtemp = regress.to_numpy(copy=True)
X = np.vstack([np.ones((len(Xtemp))), Xtemp[:,1], Xtemp[:,2], \
               Xtemp[:,1]*Xtemp[:,2] ,Xtemp[:,1]**2, Xtemp[:,2]**2]).T
bmat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

""" Regression über statmodels """

model = ols("Cp ~ rF + T + I(rF*T) + I(rF**2) + I(T**2)" , regress).fit()
print(model.summary())
b = model.params

""" Darstellung der Regressionsfunktion zusammen mit Originaldaten """

rFplot = np.linspace(0,100,10)
Tplot = np.linspace(0,100,10)
rFplotv, Tplotv = np.meshgrid(rFplot,Tplot)
Cpplotv = b[0] + b[1]*rFplotv + b[2]*Tplotv \
        + b[3]*rFplotv*Tplotv + b[4]*rFplotv**2 + b[5]*Tplotv**2

fig4 = plt.figure(4, figsize=(6, 4))
fig4.suptitle('')
ax1 = fig4.subplots(1,1)
ax1 = Axes3D(fig4)
ax1.plot_surface(Tplotv, rFplotv, Cpplotv, cmap=cm.hsv, linewidth=0, antialiased=False)
ax1.scatter(regress['T'], regress['rF'], regress['Cp']+5 ,Color='b', alpha=1)
plt.title('Vergleich Messpunkte und Regression')
ax1.set_xlabel('$T$ / °C')
ax1.set_ylabel('$rF$ / %')
ax1.set_zlabel('$C$ / pF')

""" Reststreuungsanalyse """

fig5 = plt.figure(5, figsize=(12, 4))
fig5.suptitle('Reststreuungsanalyse')
ax1, ax2 = fig5.subplots(1,2)
ax1.plot(regress['rF'],model.resid,'bo')
ax1.axis([0,100,-2,2])
ax1.set_xlabel('Relative Feuchte $rF$ / %');
ax1.set_ylabel('Residuen $Cp$ / pF');  
ax1.grid(True)
ax2.plot(regress['T'],model.resid,'bo')
ax2.axis([0,100,-2,2])
ax2.set_xlabel('Temperatur $T$ / °C');
ax2.set_ylabel('Residuen $Cp$ / pF');  
ax2.grid(True)

""" Reduktion der Regressionsfunktion über statmodels """

model = ols("Cp ~ rF + T + I(rF*T) + I(rF**2) + I(T**2)" , regress).fit()
print(model.summary())
model = ols("Cp ~ rF + T + I(rF*T) + I(T**2)" , regress).fit()
print(model.summary())
model = ols("Cp ~ rF + T + I(T**2)" , regress).fit()
print(model.summary())
model = ols("Cp ~ rF + I(T**2)" , regress).fit()
print(model.summary())
b = model.params

""" Darstellung der Regressionsfunktion nach Reduktion """

Cpplotv = b[0] + b[1]*rFplotv + b[2]*Tplotv**2
# Cpplotv = b[0] + b[1]*rFplotv + b[2]*Tplotv + b[3]*Tplotv**2

fig6 = plt.figure(6, figsize=(6, 4))
fig6.suptitle('')
ax1 = fig6.subplots(1,1)
ax1 = Axes3D(fig6)
ax1.plot_surface(Tplotv, rFplotv, Cpplotv, cmap=cm.hsv, linewidth=0, antialiased=False)
ax1.scatter(regress['T'], regress['rF'], regress['Cp']+5 ,Color='b', alpha=1)
plt.title('Regression nach Reduktion')
ax1.set_xlabel('$T$ / °C')
ax1.set_ylabel('$rF$ / %')
ax1.set_zlabel('$C$ / pF')

""" Reststreuungsanalyse nach Reduktion """

fig7 = plt.figure(7, figsize=(12, 4))
fig7.suptitle('Reststreuungsanalyse nach Reduktion')
ax1, ax2 = fig7.subplots(1,2)
ax1.plot(regress['rF'],model.resid,'bo')
ax1.axis([0,100,-2,2])
ax1.set_xlabel('Relative Feuchte $rF$ / %');
ax1.set_ylabel('Residuen $Cp$ / pF');  
ax1.grid(True)
ax2.plot(regress['T'],model.resid,'bo')
ax2.axis([0,100,-2,2])
ax2.set_xlabel('Temperatur $T$ / °C');
ax2.set_ylabel('Residuen $Cp$ / pF');  
ax2.grid(True)





