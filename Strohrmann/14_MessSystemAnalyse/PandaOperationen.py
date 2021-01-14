# -*- coding: utf-8 -*-

""" DFSS: Umgang mit Pandas Objektn
Kommentare in deutscher Sprache


Created on Sat Jan  2 15:13:47 2021
@author: stma0003
"""

from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


""" Laden eines Datensatz und Erstellen eines Dataframe, das eigentlich für 
die  Mess-System-Analyse verwendet wird, Dataframe hat 
- wie jedes Dataframe einen Index, 
- eine Teilnummer Part
- einen Prüfer Appraiser 
- die Nummer der Messung und 
- einen Messwert Value
Vorteil des Datenformats Dataframe ist, dass die Variable auf für viele 
unterschiedliche Aufgaben verwendet werden kann. Dazu muss der Gebrauch 
bekannt sein.
"""
data = loadmat('variation_proc2')
df = pd.DataFrame({'Part': np.tile(np.arange(0, 10, 1), 6),
                   'Measurement': np.tile(np.repeat([1, 2], 10), 3),
                   'Appraiser': np.repeat(['A', 'B', 'C'], 20),
                   'Value': data["data"].reshape(-1, order='C')})
print()
print(df)

""" Dataframes können über ihren Index adressierrt werden. Dabei wird 
der Dataframe zeilenweise angesprochen
"""
print()
print("Erste Zeile des Dataframes als Index und als Zeile mit Index")
print(df.index[0])
print(df.values[0])

""" Dataframes können über ihren Spaltennamen adressierrt werden. Dabei wird 
der Dataframe spaltenweise angesprochen, der Index bleibt erhalten
"""
print()
print("Zugriff auf die Daten einer Spalte, Index bleibt erhalten ")
print(df.Appraiser)
print(df["Appraiser"]) 

""" In einem Dataframe gibt es die Möglichkeit, Bereiche durch die 
den loc-Indexer auszuschneiden, dabei können auch logische Ausdrücke
verwendet werden, es werden die entsprchenden Indizes mit angezeigt
"""
print()
print("Daten mit Appraiser B und Mesurment 2, Index bleibt erhalten ")
print(df.loc[(df.Appraiser =='B') & (df.Measurement ==2)])
print()
print("Spalte kann ausgewählt werden")
print(df.loc[(df.Appraiser =='B') & (df.Measurement ==2), ["Part", "Value"]])

""" In den bisher verwendeten Beispielen wurde ein numerischer Index 
verwendet. Der Datensatz kann aber auch so verstanden werden, dass die 
Spalten Part, Appraiser und Measurment als Indizes für die Werte Value 
genutzt werden. Das entspricht auch eher der Logik, dass die Indizes die 
Bedingung der Messung festlegen und der dazu passende Messwert erfasst wird.
Dazu kann das Datafame in ein Dataframe mit Multiindex  überführt werden.
"""
df2 = df.set_index(['Appraiser','Measurement','Part'])
print()
print("Daten mit Multiindex-Format")
print(df2)
print("Daten mit Appraiser B und Measurement 2")
print(df2.loc['B',2,:])

""" Dadurch lassen sich jetzt Abfragen erstellen. Zum Vergleich die 
Auswahl von oben: Daten mit Appraiser B und Measurment 2
"""
index = pd.MultiIndex.from_product([['A', 'B', 'C'],[1, 2],np.arange(1,11,1)])
print(index)
df3= pd.DataFrame(data["data"].reshape(-1, order='C'), index=index)
df3.index.names = ['Appraiser','Measurement','Part']

print()
print("Daten mit Multiindex-Format")
print(df3)
print("Daten mit Appraiser B und Measurement 2")
print(df3.loc['B',2,:])

""" Außerdem können die Daten ausgewertet werden, dazu stehen 
Funktionen wie mean, sum und max zur Verfügung. Es muss 
dabei angegeben werden, welcher Index erhalten bleiben soll.
"""
print()
print("Mittelwert als Funktion der Teile")
print(df3.mean(level=['Part']))
print()
print("Mittelwert als Funktion der Teile und der Messung")
print(df3.mean(level=['Measurement','Part']))

""" Der Mittelwert kann auch über eine Vorauswahl von Elementen erfolgen.
"""
print()
print("Mittelwert für Prüfer A als Funtion der Teile")
print(df3.loc['A',:,:].mean(level=['Part']))

print(df3.reset_index())








