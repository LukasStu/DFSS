from scipy.stats import norm   # normal-Verteilung
from scipy.stats import t     # t-Verteitung
from scipy.stats.stats import pearsonr
import numpy as np
import math



def corranalysa_2d (X,Y,Alpha=0.05):
    '''% Bewertung des Korrelationskoeffizienten'''
    r=np.corrcoef(X,Y)[0,1]
    N=len(X)
    c1=norm.ppf(Alpha/2,0,1)
    c2=norm.ppf((2-Alpha)/2,0,1)
    rmin= math.tanh(math.atanh(r)-(c2/np.sqrt(N-3)))
    rmax= math.tanh(math.atanh(r)-(c1/np.sqrt(N-3)))
    pvalue=pearsonr(X,Y)[1]
    
    '''Zur Signifikanzbewertung des Korrelationskoeffizienten wird ein Hypothesentest durchgeführt'''
    c1t = t.ppf(Alpha/2,N-2)
    c2t = t.ppf(1-Alpha/2,N-2)
    t0 = r*np.sqrt((N-2)/(1-r**2))
    return r,pvalue,rmin,rmax, c1t,c2t,t0



def corranalysa_2d_GV (X,Y,Alpha=0.05):
    '''Bewertung des Korrelationskoeffizienten'''
    r=np.corrcoef(X,Y)[0,1]
    N=len(X)
    c1=norm.ppf(Alpha/2,0,1)
    c2=norm.ppf((2-Alpha)/2,0,1)
    rmin= math.tanh(math.atanh(r)-(c2/np.sqrt(N-3)))
    rmax= math.tanh(math.atanh(r)-(c1/np.sqrt(N-3)))
    pvalue=pearsonr(X,Y)[1]
    """ Grenzen für den Annahmebereich von dem Regressionskoeffizienten""" 
    rc1=math.tanh(c1/np.sqrt(N-3))
    rc2=math.tanh(c2/np.sqrt(N-3))
    '''Zur Signifikanzbewertung des Korrelationskoeffizienten wird ein Hypothesentest durchgeführt'''
    c1t = t.ppf(Alpha/2,N-2)
    c2t = t.ppf(1-Alpha/2,N-2)
    """bei berechnetem Regressionskoeffizient r0 """
    t0 = r*np.sqrt((N-2)/(1-r**2))
    '''Für einen Vergleich werden die Grenzen rC1 und rC2 in die Gleichung für die Zufallsvariable t eingesetzt und mit den Grenzen verglichen, die sich aus der t-Verteilung ergeben.'''
    trc1 = rc1*np.sqrt((N-2)/(1-rc1**2))
    trc2 = rc2*np.sqrt((N-2)/(1-rc2**2))
    return r,pvalue,c1,c2,rmin,rmax,rc1,rc2,c1t,c2t,t0,trc1,trc2



def corranalysa_3d (X,Y,Z,Alpha=0.05):
    '''% Bewertung des Korrelationskoeffizienten'''
    N=len(X)
    
    r_xy=np.corrcoef(X,Y)[0,1]; r_xz=np.corrcoef(X,Z)[0,1]; r_yz=np.corrcoef(Y,Z)[0,1]
    
    R_Matrix=[[1,r_xy,r_xz],[r_xy,1,r_yz],[r_xz,r_yz,1]]
    
    c1=norm.ppf(Alpha/2,0,1)
    c2=norm.ppf((2-Alpha)/2,0,1)
    
    Rmin_xy= math.tanh(math.atanh(r_xy)-(c2/np.sqrt(N-3)))
    Rmax_xy= math.tanh(math.atanh(r_xy)-(c1/np.sqrt(N-3)))

    Rmin_xz= math.tanh(math.atanh(r_xz)-(c2/np.sqrt(N-3)))
    Rmax_xz= math.tanh(math.atanh(r_xz)-(c1/np.sqrt(N-3)))

    Rmin_yz= math.tanh(math.atanh(r_yz)-(c2/np.sqrt(N-3)))
    Rmax_yz= math.tanh(math.atanh(r_yz)-(c1/np.sqrt(N-3)))
    
    Rmin_Matrix=[[1,Rmin_xy,Rmin_xz],[Rmin_xy,1,Rmin_yz],[Rmin_xz,Rmin_yz,1]]
    
    Rmax_Matrix=[[1,Rmax_xy,Rmax_xz],[Rmax_xy,1,Rmax_yz],[Rmax_xz,Rmax_yz,1]]
    
    pvalue_Matrix=[[1,pearsonr(X,Y)[1],pearsonr(X,Z)[1]],[pearsonr(X,Y)[1],1,pearsonr(Y,Z)[1]],[pearsonr(X,Z)[1],pearsonr(Y,Z)[1],1]]
    
    return R_Matrix,pvalue_Matrix,Rmin_Matrix,Rmax_Matrix
