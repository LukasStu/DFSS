# -*- coding: utf-8 -*-
"""
Beispiel für die Auswertung eines Datensatzes mit Python,
als Vergleich zu MATLAB
"""

""" Bibliotheken importieren"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.io import loadmat

"""Einlesen und Umsortieren der Daten aus dem .mat-file"""
data = loadmat('Klebermenge')['data']
#X = np.array(data).reshape(data.shape[0]*data.shape[1])
X = np.array(data).reshape(-1)

#print(data.shape)
#print(X.shape)

"""Bestimmung Datenumfang"""
X_min = np.amin(X)
X_max = np.amax(X)
N = X.shape[0]


"""Berechnen der Lagekennwerte Mittelwert und Median"""
X_mean = np.mean(X)
print(' ')
print('Arithmetischer Mittelwert: ', X_mean)
X_med = np.median(X)
print('Median:', X_med )


""" Berechnen der Streuungskennwerte Stadardabweichung, Varianz und """
""" Inter-Quartil-Range, bei Varianz muss die Anzahl der Freiheitsgrade"""
""" auf N - 1 angepasst werden"""
X_var = np.var(X, ddof=1)
print(' ')
print('Varianz: ', X_var)
X_std = np.std(X,ddof=1)
print('Standardabweichung: ', X_std)
X_iqr = np.quantile(X,0.75) - np.quantile(X,0.25)
print('Inter-Quartil-Range: ', X_iqr)


"""Berechnen der Schiefe"""
X_skew_mom = skew(X)
print(' ')
print('Momentenkoeffizient der Schiefe:', X_skew_mom )
X_skew_qua = (np.quantile(X,0.75) - 2*np.quantile(X,0.5) + np.quantile(X,0.25))/X_iqr
print('Quartilkoeffizient der Schiefe:', X_skew_qua )


""" Grafische Darstellung der einzelnen Messwete als Streudiagramm """
fig = plt.figure(1, figsize=(12, 4))
ax1, ax2 = fig.subplots(1,2)
n = np.arange(1,N+1)
ax1.plot(n,X, 'bo', Linewidth = 2, label = 'Stichproben')
ax1.grid(True, which='both', axis='both', linestyle='--')
ax1.axis([1, N, 49, 53])
ax1.legend(loc='upper right')
ax1.set_xlabel('Stichprobe n')
ax1.set_ylabel('Masse m / mg')

""" Erstellen eines Boxplt """
ax2.boxplot(X)
ax2.grid(True, which='both', axis='both', linestyle='--')
ax2.set_xlabel('Messreihe ')
ax2.set_ylabel('Klebermenge m / mg')
ax2.axis([0, 2, 48, 54])


""" Berechnung der absoluten und relativen Häufigkeit sowie der absoluten"""
""" und relativen Summenhäufigkeit"""
X_freq, Klassengrenzen = np.histogram(X, bins=np.arange(np.floor(X_min)-0.5, np.ceil(X_max)+1.5))
X_rel_freq = X_freq/N
X_sum_freq = np.cumsum(X_freq)
X_rel_sum_freq = X_sum_freq/N
Klassenmitten = np.arange(np.floor(X_min),np.ceil(X_max)+1)


#""" Generieren einer Tabelle in Pandas und Ausgabe der Tabelle"""
#Tabelle = pd.DataFrame({'Gruppenwert':Klassenmitten, 
#                        'Absolute Häufigkeit hA(x)':X_freq,
#                        'RelativeHaeufigkeit h(x)':X_rel_freq,
#                        'Absolute Summenhäufigkeit HA(x)':X_sum_freq,
#                        'Relative Summenhäufigkeit H(x)':X_rel_sum_freq})
#print(' ')
#print(Tabelle)



""" Grafische Darstellung der relativen Häufigkeiten als Histogramm """
fig = plt.figure(2, figsize=(12, 4))
ax1, ax2 = fig.subplots(1,2)
ax1.hist(X, Klassengrenzen, histtype='bar' , color='b', weights=np.ones(N)/N, rwidth=1)
ax1.grid(True, which='both', axis='both', linestyle='--')
ax1.set_xlabel('Klebermenge m / mg')
ax1.set_ylabel('Relative Häufigkeit h(m)')
ax1.axis([48, 54, 0, 0.6])

""" Grafische Darstellung der relativen Summenhäufigkeit """
Xsort = np.append(np.append(48,np.sort(X)),54)
Psum = np.append(np.append([0,],np.arange(1,N+1)/N),1)
ax2.step(Xsort,Psum, color='b', where='post', linewidth=2)
ax2.grid(True, which='both', axis='both', linestyle='--')
ax2.set_xlabel('Klebermenge m / mg')
ax2.set_ylabel('Relative Summenhäufigkeit H(m)')
ax2.axis([48, 54, 0, 1])














