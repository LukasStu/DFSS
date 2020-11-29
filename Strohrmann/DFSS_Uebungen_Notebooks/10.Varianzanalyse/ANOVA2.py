import numpy as np 
import pandas as pd
from scipy.stats import f




def ssd(ser):
    '''
    Function ssd(): computes the sum of squared deviates for a Series object
    
    > Input parameters:
    - ser: the Series object
    
    > Returns:
    - The sum of squared deviates computed as Σ(x)**2 - ((Σx)**2)/N  
    '''
    ser.dropna(axis=0, inplace=True)     # Clear Series from null values 'in place'
    s1 = pow(ser,2).sum()
    s2 = pow(ser.sum(),2) / ser.size
    return s1-s2



def ssd_df(indf):
    '''
    Function ssd_df(): computes the ssd: Σ(x)**2 - ((Σx)**2)/N factor for a DataFrame object as a whole
      
    > Input parameters:
    - df: the DataFrame object
    It is ALWAYS assumed that the 1st column (0-index) is the R-factor and is ommited from computations
    
    > Returns a tuple consisting of:
    - ss_n_all = The ssd: Σ(x)**2 - ((Σx)**2)/N factor 
    - n_all = The size of DataFrame data included in the computation 
    '''
    
    n_all = sumx = sumx2 = 0
    for i in range(1,len(indf.columns)):
        ser = indf.iloc[:,i].dropna()
        sumx += ser.sum()
        sumx2 += pow(ser,2).sum()
        n_all += ser.size        
    sumx_sqed = pow(sumx,2)
    ss_n_all = sumx2 - (sumx_sqed / n_all)
    return ss_n_all, n_all



def ssd_df_rc(df, axis=0):
    '''
    Function ssd_df_rc(): computes the sum of squared deviates for the two way anova rows or columns  
    This sum is computed as Σ((Σ(x)**2)/Ν) - ((Σx_all)**2)/N_all
    It is ALWAYS assumed that the 1st column (0-index) is the R-factor and is ommited from computations
    
    > Input parameters:
    - df: the DataFrame object
    - axis=0 column-wise (working on columns data)
    - axis=1 row-wise (working on rows data) 
    
    > Returns:
    - The sum of squared deviates for the anova rows or columns of the input DataFrame
    '''
    
    ss_n_sum = 0
    ss_n_all = 0
    
    if axis == 0:
        # Compute the ss_n_sum quantity considering each SEPARATE Column in df 
        for i in range(1,len(df.columns)):
            c_ser = df.iloc[:,i].dropna()
            ss_n_sum += pow(c_ser.sum(),2) / c_ser.size
    
    elif axis == 1:
        r_factor = df.columns[0]
        anv_groups = df.groupby(r_factor)
        for symb, gp in anv_groups:
            # Compute the ss_n_sum quantity considerint each SEPARATE Row in df
            # Rows in df are ADDED columns in each anv_groups
            n_all = sumx = 0
            for i in range(1,len(gp.columns)):
                ser = gp.iloc[:,i].dropna()
                sumx += ser.sum()
                n_all += ser.size        
            ss_n_sum += pow(sumx,2) / n_all

    else:
        print('axis undefined in ssd_df_rc()')
        return 

    # Compute the ((Σx_all)**2)/N_all factor for ALL data in the DataFrame
    n_all = sumx = sumx_p2 = 0
    for i in range(1,len(df.columns)):
        ser = df.iloc[:,i].dropna()
        sumx += ser.sum()
        n_all += ser.size        
    sumx_sqed = pow(sumx,2)
    ss_n_all = sumx_sqed / n_all
    
    return ss_n_sum - ss_n_all
    


def ptl_anova2(inframe):                  
    '''
    Function:  ptl_anova2() for performing TWO way anova on input data
    
    > Input parameters:
    - inframe: pandas DataFrame with data groups as follows: 
    --- Column 0: the R-factor determining grouping
    --- Other Columns: Data grouped according to C-factor

    > Returns:
    - F: the F statistic for the input data
    - p: the p probability for statistical significance    
 
    '''
    # Detecting the shape of inframe:
    rows, cols = inframe.shape
    
    # Detecting the R x C anova design 
    c = len(inframe.columns)-1
    r_factor = inframe.columns[0]
    anv_groups = inframe.groupby(r_factor)
    r = len(anv_groups)

    # Computing ss_t and n_t with the ss_df() function
    ss_t, n_t = ssd_df(inframe)
    
    # Computing ss_wg with groupby.agg()
    ss_wg = 0
    ss_wg_cells = anv_groups.agg(ssd)
    ss_wg = ss_wg_cells.sum().sum()
     
    # Compute ss_bg by subtracking ss_wg from ss_t
    ss_bg = ss_t - ss_wg
       
    # ADDITIONAL (compared to One-way) computations in Two way ANOVA: ss_r, ss_c, ss_int
    # a) ss_c
    ss_c = ssd_df_rc(inframe, axis=0)
    
    # b) ss_r
    ss_r = ssd_df_rc(inframe, axis=1)
    
    # c) ss_int
    ss_int =  ss_bg - ss_r - ss_c 
      
    # degrees of freedom
    df_t = n_t - 1
    df_bg = r*c - 1
    df_wg = df_err = n_t - r*c
    df_r = r - 1 
    df_c = c - 1
    df_int = df_r * df_c
    
    # Mean Square (MS) factors
    ms_r = ss_r / df_r
    ms_c = ss_c / df_c
    ms_int = ss_int / df_int
    ms_wg = ms_err = ss_wg / df_wg
            
    # F, p
    F_r = ms_r / ms_err
    p_r = f.sf(F_r, df_r, df_err, loc=0, scale=1)
    
    F_c = ms_c / ms_err
    p_c = f.sf(F_c, df_c, df_err, loc=0, scale=1)
    
    F_int = ms_int / ms_err
    p_int = f.sf(F_int, df_int, df_err, loc=0, scale=1)
    
    # Printouts
    print('  bg: \t SS = {:9.3f}, \t df = {:3d}'.format(ss_bg, df_bg))
    print('Rows: \t SS = {:9.3f}, \t df = {:3d}, \t ms = {:9.3f}, \t F = {:9.3f}, \t p = {:8.4f}'\
          .format(ss_r, df_r, ms_r, F_r, p_r))
    print('Cols: \t SS = {:9.3f}, \t df = {:3d}, \t ms = {:9.3f}, \t F = {:9.3f}, \t p = {:8.4f}'\
          .format(ss_c, df_c, ms_c, F_c, p_c))
    print(' Int: \t SS = {:9.3f}, \t df = {:3d}, \t ms = {:9.3f}, \t F = {:9.3f}, \t p = {:8.4f}'\
          .format(ss_int, df_int, ms_int, F_int, p_int))      
    print('  wg: \t SS = {:9.3f}, \t df = {:3d}, \t ms = {:9.3f}'.format(ss_wg, df_wg, ms_wg))
    print('TOTL: \t SS = {:9.3f}, \t df = {:3d}'.format(ss_t, df_t))
    
    
    return