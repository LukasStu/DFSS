# -*- coding: utf-8 -*-

""" Musterlösung zur Klausur Design For Six Sigma SS2018
Umsetzung in Python im WS 2020/21

Update on sun Jan 24 2021
@author: stma0003
"""

from scipy import stats
from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm

def conf_pred_band_ex(_regress_ex, _poly, _model, alpha=0.05):
    """ Function calculates the confidence and prediction interval for a
    given multivariate regression function poly according to lecture DFSS,
    regression parameters are already determined in an existing model,
    identical polynom is used for extrapolation

    Parameters
    ----------
    regress_ex : DataFrame
        Extended dataset for calculation.
    poly : OLS object of statsmodels.regression.linear_model modul
        definition of regression model.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Model parameters.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    lconf_ex : Series
        Distance of confidence limit to mean regression.
    lpred_ex : Series
        Distance of prediction limit to mean regression.
    """

    # ols is used to calculte the complets vector x_0 of input variables
    poly_ex = ols(_poly.formula, _regress_ex)
    x_0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x_0, np.dot(_poly.normalized_cov_params, x_0.T))
    c_1 = stats.t.isf(alpha/2, _model.df_resid)
    lconf_ex = c_1*np.sqrt(np.diag(d)*_model.mse_resid)
    lpred_ex = c_1*np.sqrt((1+np.diag(d))*_model.mse_resid)

    return lconf_ex, lpred_ex


""" Aufgabenteil a: Hypothesentest auf gleichen Mittelwert """

# Load and format data
data = loadmat('Signifikanz')
mass_hyp = pd.DataFrame({'m1': data["m1"].reshape(-1),
                         'm2': data["m2"].reshape(-1)})

# t-test due to unknown variance
hypo_test = stats.ttest_ind(mass_hyp["m1"], mass_hyp["m2"])
print("")
print("Hypothesentest auf Abweichung mit p-value = ",
      round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")


""" Aufgabenteil b: Varianzanalyse """

# Format data to pandas structure
mass_ols = pd.DataFrame({'n': np.repeat([75, 125], 8),
                         'm': pd.concat([mass_hyp["m1"], mass_hyp["m2"]])})

# ANOVA Tabelle
model = ols('m ~ C(n)', data=mass_ols).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
print("")
print("Anova auf Abweichung mit p-value = ",
      round(float(anova1["PR(>F)"]["C(n)"]), 4))
if anova1["PR(>F)"]["C(n)"] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")


""" Aufgabenteil c: Bewertung Regressionsfunktion """

# Regression function of conductivity as a function of temperature
poly = ols("m ~ n", mass_ols)
model = poly.fit()
print("")
print("Prüfung Signifikanz des linearen Terms mit p-value = ",
      round(float(model.pvalues["n"]), 4))
if model.pvalues["n"] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")


""" Aufgabenteil d: Vergleich der Ergebnisse """

print("")
print("Alle Verfahren führen zu dem Ergebnis, dass die beiden Stichproben ")
print("nicht signifikant voneinander abweichen. Interesant ist, dass alle")
print("Tests sogar alle p-Values denselben Wert aufweisen.")


""" Aufgabenteil e: Boxplot """

# Visualization
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)
boxplot = mass_hyp.boxplot(column=['m1', 'm2'], ax=ax1)
ax1.axis([0, 3, 108.2, 108.23])
ax1.set_xlabel('Gruppe')
ax1.set_ylabel('Masse $m$ / g')
ax1.grid(True)
print("")
print("Boxen des Box-Plot überlappen, Grafik bestätigt, dass keine")
print("signifikante Abweichung vorliegt.")


""" Aufgabenteil m """

# Load and format data
data = loadmat('Tolerierung')
mass_reg = pd.DataFrame({'tz': data["TZ"].astype(np.float64).reshape(-1),
                         'm': data["m5"].reshape(-1)})
poly = ols("m ~ tz + I(tz**2)", mass_reg)
model = poly.fit()
print("")
print(model.summary())
print("")
print("Vollquadratisches Modell, alle Terme signifikant,")
print("Bestimmtheitsmaß sehr hoch.")


""" Aufgabenteil n """

# Covariance matrix of regression coefficients
beta_cov = poly.normalized_cov_params*model.mse_resid
print("")
print("Kovarianzmatrix der Regressionkoefizienten")
print("")
print(beta_cov)
print("")
print("Elemente der Kovarianzmatrix sind auch außerhalb der Hauptdiagonalen")
print("von null verschieden, Regressionskoeffizienten sind abhängig")


""" Aufgabenteil o """

# Tolerance chain
# dm = db0 +TZ_0*db1 + b10*dtz + TZ_0^2*db2 + 2*TZ_0*b20*dtz
TZ_0 = 230
TZ_SIG = 10/6
b0_var = beta_cov[0, 0]
b1_0 = model.params["tz"]
b1_var = beta_cov[1, 1]
b2_0 = model.params["I(tz ** 2)"]
b2_var = beta_cov[2, 2]
b0b1_var = beta_cov[0, 1]
b0b2_var = beta_cov[0, 2]
b1b2_var = beta_cov[1, 2]
mass_tol = 6*np.sqrt(b0_var
                     + TZ_0**2 * b1_var
                     + b1_0**2 * TZ_SIG**2
                     + TZ_0**4 * b2_var
                     + 4*TZ_0**2 * b2_0**2*TZ_SIG**2
                     + 2*TZ_0 * b0b1_var
                     + 2*TZ_0**2 * b0b2_var
                     + 2*TZ_0**3 * b1b2_var)
print("")
print('Toleranz der Masse Tm =', round(mass_tol, 4))


""" Aufgabenteil p """

# Single tolerance cause
mass_tol_causes = \
    pd.DataFrame({'portion': [(4*TZ_0**2*b2_0**2 + b1_0**2)*TZ_SIG**2,
                              b0_var,
                              TZ_0**2 * b1_var,
                              TZ_0**4 * b2_var,
                              2*TZ_0 * b0b1_var,
                              2*TZ_0**2 * b0b2_var,
                              2*TZ_0**3 * b1b2_var]},
                 index=['TZ', 'b0', 'b1', 'b2', 'b0b1', 'b0b2', 'b1b2'])

# Tolerance groups
mass_tol_groups = \
    pd.DataFrame({'portion': [(4*TZ_0**2*b2_0**2 + b1_0**2)*TZ_SIG**2,
                              b0_var
                              + TZ_0**2 * b1_var
                              + TZ_0**4 * b2_var
                              + 2*TZ_0 * b0b1_var
                              + 2*TZ_0**2 * b0b2_var
                              + 2*TZ_0**3 * b1b2_var]},
                 index=['TZ', 'Reg'])

# Visualization
fig1 = plt.figure(2, figsize=(12, 4))
fig1.suptitle('')
ax1, ax2 = fig1.subplots(1, 2)
mass_tol_causes.plot.bar(ax=ax1, legend=None)
ax1.axis([-1, 7, -7, 7])
ax1.set_xlabel('Toleranzursache')
ax1.set_ylabel('Masse m / g')
ax1.grid(True)
mass_tol_groups.plot.bar(ax=ax2, legend=None)
ax2.axis([-1, 2, 0, 0.1])
ax2.set_xlabel('Toleranzursache')
ax2.set_ylabel('Masse m / g')
ax2.grid(True)

print("")
print("Kovarianz der Regressionskoeffizienten führt zur Kompensation der")
print("durch die einzelnen Koeffizienten hervorgerufenen Toleranzanteile. ")
print("")
print("")
print("")
