# time series analysis function
from scipy import stats
import xarray as xr
import numpy as np

import scipy.stats
import numpy as np 
import sys
import matplotlib.pyplot as plt
# from scipy.stats import betai
from scipy.stats import t
# from module.check_match import checkmatch_time
from scipy import signal
from numpy import linalg as la
from scipy import stats


### granger causality example
"""
from statsmodels.tsa.stattools import grangercausalitytests
gstat = grangercausalitytests(df_ts[['HT_180','ZOS_WE_diff']], maxlag=8)

"""

def ts_mean_conf(ts,stTconfint=0.99):
    ds_ts = xr.Dataset()
    ds_ts['ts'] = ts.squeeze()

    
    ds_ts = ds_ts.where(ds_ts.ts.notnull(),drop=True)

    std_err = ds_ts.ts.std()/np.sqrt(len(ds_ts.ts))
    
    
    ### calculate confidence interval 
    # calculate the error bar base on the number of standard error
    # the number related to dist. percentage is derived base on Students's T
    # distribution
    dof = len(ds_ts.ts)-1
    alpha = 1.0-stTconfint
    nstd = stats.t.ppf(1.0-(alpha/2.0),dof)  # 2-side
    mean_conf = nstd*std_err


    return mean_conf


def ts_seg_slope(ts,syear,fyear,stTconfint=0.95):
    ds_ts = xr.Dataset()
    ds_ts['ts'] = ts.squeeze()
    
    nyear = len(ts)/12
    yeararray = 1/24.+np.arange(12.*nyear)/12.
    ds_ts['yeardate'] = xr.DataArray(yeararray,
                                     dims='time',
                                     coords={'time':ts.time}) 
    
    ds_ts = ds_ts.where(ds_ts.ts.notnull(),drop=True)
    slope_tot, intercept, r_value, slope_tot_pval, std_err\
       =stats.linregress(ds_ts.yeardate.values,ds_ts.ts.values)
    
    ### calculate confidence interval 
    # calculate the error bar base on the number of standard deviation
    # the number of standard deviation is derived base on Students's T
    # distribution
    dof = len(ds_ts.ts)-1
    alpha = 1.0-stTconfint
    nstd = stats.t.ppf(1.0-(alpha/2.0),dof)  # 2-side
    slope_tot_conf = nstd*std_err


    ds_ts = ds_ts.where((ds_ts['time.year']>=syear)&\
                        (ds_ts['time.year']<=fyear)\
                        ,drop=True)  
    slope, intercept, r_value, slope_pval, std_err\
       =stats.linregress(ds_ts.yeardate.values,ds_ts.ts.values)
    
    ### calculate confidence interval 
    # calculate the error bar base on the number of standard error
    # the number of standard deviation is derived base on Students's T
    # distribution
    dof = len(ds_ts.ts)-1
    alpha = 1.0-stTconfint
    nstd = stats.t.ppf(1.0-(alpha/2.0),dof)  # 2-side
    slope_conf = nstd*std_err

    return slope, slope_conf, slope_tot, slope_tot_conf


def lead_lag_corrcoef_conf(ts1,ts2,leadlag=12):
    # input numpy array 
    
    corrcoef = []
    aic = []
    corrconf = []
    
    ind3 = np.where(~np.isnan(ts1+ts2))[0]
    ts1 = ts1[ind3]
    ts2 = ts2[ind3]
    
    # for ts1 lead ts2
    for lag in reversed(np.arange(leadlag)+1):
        corrcoef.append(stats.linregress(ts1[:-lag],ts2[lag:])[2])
        corrconf.append(corr_conf(ts1[:-lag],ts2[lag:],nmember=1000,conf_interv=95))
        aic.append(cal_aic(ts1[:-lag],ts2[lag:]))
        
    # for no lead lag
    corrcoef.append(stats.linregress(ts1[:],ts2[:])[2])
    corrconf.append(corr_conf(ts1[:],ts2[:],nmember=1000,conf_interv=95))
    aic.append(cal_aic(ts1[:],ts2[:]))
    
    
    # for ts1 lag ts2
    for lag in np.arange(leadlag)+1:
        corrcoef.append(stats.linregress(ts1[lag:],ts2[:-lag])[2])
        corrconf.append(corr_conf(ts1[lag:],ts2[:-lag],nmember=1000,conf_interv=95))
        aic.append(cal_aic(ts1[lag:],ts2[:-lag]))
        
    corrcoef = np.array(corrcoef)
    corrconf = np.array(corrconf)
    aic = np.array(aic)
    leadlag_array = (np.arange(leadlag*2+1)-leadlag)
    max_ind = np.where(np.abs(corrcoef)==np.max(np.abs(corrcoef)))[0][0]

                 
    return corrcoef, leadlag_array, max_ind, aic, corrconf

def lead_lag_corrcoef(ts1,ts2,leadlag=12):
    # input numpy array 
    
    corrcoef = []
    aic = []
   
#     ind1 = np.where(~np.isnan(ts1))[0]
#     ind2 = np.where(~np.isnan(ts2))[0]
    ind3 = np.where(~np.isnan(ts1+ts2))[0]

    ts1 = ts1[ind3]
    ts2 = ts2[ind3]
    
    # for ts1 lead ts2
    for lag in reversed(np.arange(leadlag)+1):
        corrcoef.append(stats.linregress(ts1[:-lag],ts2[lag:])[2])
#         aic.append(cal_aic(ts1[:-lag],ts2[lag:]))
        
    # for no lead lag
    corrcoef.append(stats.linregress(ts1[:],ts2[:])[2])
#     aic.append(cal_aic(ts1[:],ts2[:]))
    
    
    # for ts1 lag ts2
    for lag in np.arange(leadlag)+1:
        corrcoef.append(stats.linregress(ts1[lag:],ts2[:-lag])[2])
#         aic.append(cal_aic(ts1[lag:],ts2[:-lag]))
        
    corrcoef = np.array(corrcoef)
#     aic = np.array(aic)
    leadlag_array = (np.arange(leadlag*2+1)-leadlag)
    max_ind = np.where(np.abs(corrcoef)==np.max(np.abs(corrcoef)))[0][0]

                 
    return corrcoef, leadlag_array, max_ind, aic


def cal_aic(ts1,ts2):
    # input numpy array 
    
    # when using ts1 to predict ts2
    ind1 = np.where(~np.isnan(ts1))[0]
    ind2 = np.where(~np.isnan(ts2))[0]
    
    ts1 = ts1[ind1]
    ts2 = ts2[ind2]
    
    regress = stats.linregress(ts1,ts2)
    model = regress[0]*ts1+regress[1]
    prob, xbin = np.histogram(model, bins=10)
    prob = prob/len(ts1)
    max_prob = np.max(prob)
    max_xbin = np.argmax(prob)
    
    aic = 2*1-2*np.log(max_prob)
    
    return aic
    

    
def remove_mean_trend(da_ts):
    # input xarray.DataArray
    
    da_ts = da_ts.where(da_ts.notnull(),drop=True)

    regress = stats.linregress(np.arange(len(da_ts.values)),da_ts.values)
    model = regress[0]*np.arange(len(da_ts.values))+regress[1]
    
    da_ts = da_ts-model
    
    return da_ts
    
def pick_pos_neg_seg(ts,initime=None,endtime=None):
    # integrate segment that is above/below zeros
    
    if initime is not None and endtime is not None :
        ts = ts.where((ts.time>=initime) & (ts.time<=endtime))
    
    # calculate dt
    dt = ts.copy().squeeze()
    dt.values[:-1] = ts.time.diff(dim='time').values/np.timedelta64(1, 's')     # second
    dt.values[-1] = (ts.time[-1]-ts.time[-2]).values/np.timedelta64(1, 's')    # second
    
    # seperate above/below zeros
    ts_pos=ts.where(ts>0.)
    ts_neg=ts.where(ts<0.)
    
    # calculate positive integration
    ini_tt = 0
    seg_time_index = []
    sel_pos_seg = []
    sel_pos_seg_sum = []

    for tt,time in enumerate(ts_pos.time.values):
        seg_sum = ts_pos[ini_tt:tt].sum().values
        if seg_sum > 0.:
            seg_time_index.append(tt-1)
            if seg_sum == ts_pos[ini_tt:tt-1].sum().values :
                seg_time_index.pop(-1)
                sel_pos_seg.append(ts_pos[seg_time_index])
                sel_pos_seg_sum.append((ts_pos[seg_time_index]*dt).sum().values)  #PetaJoule
                ini_tt = tt
                seg_time_index = []

    # calculate negative integration
    ini_tt = 0
    seg_time_index = []
    sel_neg_seg = []
    sel_neg_seg_sum = []

    for tt,time in enumerate(ts_neg.time.values):
        seg_sum = ts_neg[ini_tt:tt].sum().values
        if seg_sum < 0.:
            seg_time_index.append(tt-1)
            if seg_sum == ts_neg[ini_tt:tt-1].sum().values :
                seg_time_index.pop(-1)
                sel_neg_seg.append(ts_neg[seg_time_index])
                sel_neg_seg_sum.append((ts_neg[seg_time_index]*dt).sum().values)  #PetaJoule
                ini_tt = tt
                seg_time_index = []
                
                

    
    
                
                
    return sel_pos_seg,sel_pos_seg_sum,sel_neg_seg,sel_neg_seg_sum
                
                


def z_score(x,dmonth):
    z=np.zeros(len(x))+np.float('nan')
    for i in np.arange(12)+1:
        rem=np.zeros(len(x))+np.float('nan')
        for j in range(len(dmonth)):
            rem=np.remainder(dmonth[j],np.float(i))
        #print rem
        ind1=np.where(rem<1E-10)
        z[ind1]=(x[ind1]-np.mean(x[ind1]))/np.var(x[ind1])

    return z        

def z_score(dmonth,x):
    t=np.remainder(dmonth,12)
    t[np.where(t==0.)]=12
    z = np.empty(len(x))*np.nan
    lenz = len(x)

    for i in range(1,13):
        mon = t==i
        aa = x[mon]
        xmean = np.nanmean(aa)
        xstd = np.nanstd(aa)
        z[mon] = (aa - xmean)/xstd
    return z        

def part_r(z1,z2,zc):
    """
    coded by Geruo A
    """
    idx = ~np.isnan(zc) & ~np.isnan(z1)
    slope, interc, r_val, p_val, std_err = stats.linregress(zc[idx],z1[idx])
    res1 = z1 - zc*slope - interc

    idx = ~np.isnan(zc) & ~np.isnan(z2)
    slope, interc, r_val, p_val, std_err = stats.linregress(zc[idx],z2[idx])
    res2 = z2 - zc*slope - interc
   
#     idx = ~np.isnan(res1) & ~np.isnan(res2)
#     slope, interc, r_val, p_val, std_err = stats.linregress(res1[idx],res2[idx])
#     res = res2 - res1*slope - interc
#     print "check: ", slope, interc
#     plt.plot(res1, res,'bo')

    idx = ~np.isnan(res1) & ~np.isnan(res2)
    corr, pval = stats.pearsonr(res1[idx],res2[idx])
    return corr, pval, res1, res2

#def corrcoef(A1,A2):
#    #matrix=np.zeros([len(A1),2])
#    #matrix[:,0]=A1
#    #matrix[:,1]=A2
#    matrix=[A1,A2]
#
#    r = np.corrcoef(matrix)
#    rf = r[np.triu_indices(r.shape[0], 1)]
#    df = matrix.shape[1] - 2
#    ts = rf * rf * (df / (1 - rf * rf))
#    pf = betai(0.5 * df, 0.5, df / (df + ts))
#    p = np.zeros(shape=r.shape)
#    p[np.triu_indices(p.shape[0], 1)] = pf
#    p[np.tril_indices(p.shape[0], -1)] = pf
#    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
#    return r, p


def partial_corrcoef(A1o,A2o,Bo):
    """
    Caution!!!!! keep all time series at the same length covering the exact same range of time
    steps:
     A1 is the time series of interest of correlation between A1 and A2 but also affected by B
     A2 is the time series of interest of correlation between A1 and A2 but also affected by B
     B is the time series one want to "partial out" from A1 and A2 and set to constant
     the method of partial out B is to regress A1 onto B and A2 onto B
     the residual of A1 will then without the influence of B (same for A2)
     then the correlation is calculated bewteen residual A1 and A2
    """

    ind=~np.isnan(A1o) & ~np.isnan(A2o) & ~np.isnan(Bo)
    A1=A1o[ind]
    A2=A2o[ind]
    B=Bo[ind]


    ndata=len(A1)

    DM = np.zeros([ndata,2])
    DM[:,0] = 1.
    DM[:,1] = B[:]
    # reshape the A1 A2 to column
    A1=np.reshape(A1,[A1.shape[0],1])
    A2=np.reshape(A2,[A2.shape[0],1])
    term1=np.dot(np.transpose(DM),DM)     #(DM'DM)
    detterm1=la.det(term1)
    if detterm1 == np.float64(0.) :
        print("Error64: (DM'DM) do not have inversion")
        return 
    term1_1=la.inv(term1)                 #(DM'DM)^-1
       
    term2=np.dot(np.transpose(DM),A1)     #(DM'A1)
    beta_a1=np.dot(term1_1,term2)         #(DM'DM)^-1(DM'A1)

    term2=np.dot(np.transpose(DM),A2)     #(DM'A2)
    beta_a2=np.dot(term1_1,term2)         #(DM'DM)^-1(DM'A2)


    # calculate residual
    A1res=A1[:,0]-beta_a1[0]-beta_a1[1]*B[:]
    A2res=A2[:,0]-beta_a2[0]-beta_a2[1]*B[:]

    # calculate the coefficient 
    r,p=stats.pearsonr(A1res,A2res)
    return r,p,A1res,A2res




def corr_conf(ts1,ts2,nmember=1000,conf_interv=95): 
    """
    The correlation of the two time series would be affected by the autocorrelation increase 
    when one applied a low pass filter or moving window on the time series. To estimate the 
    static significancy of the correlation, EBISUZAKI 1997 establish a method to generate 
    large number of synthetic time series that possess the same autocorrelation characteristic
    of the original time series. One can assume that this method generates the population of a certain 
    autocorrelation character. This allowed us to calculate the confidence interval of the 
    derived correlation from a larger sample size instead of just two time series. 
    Important!!!!
        1. while using the function a significant seasonal or other time scale fluctuation 
            should be removed since it is random-phase process which would change the 
            phase of the seasonal or other time scale fluctuation.
        2. trend is best to be removed to include the accuracy of the determination 
            of confidence interval
        3. ts1 is the time series will have the nmember of synthetic time series generated 
    """
    conf_corrcoef=None
    
    #### perform FFT on ts
    nts1=ts1.shape[0]
    ts1_sp=np.fft.fft(ts1)
    #power=np.sqrt(ts1_sp.real[:]**2+ts1_sp.imag[:]**2)
    #freq=np.fft.fftfreq(nts1)

    #### generate random-phase ts
    ts1_sp_rp=np.zeros([nts1,nmember],dtype=np.complex128)
    for n in range(nmember) :
        theta=np.random.uniform(0.,2*np.pi,nts1)
        #print theta
        ts1_sp_rp[1:,n]=np.sqrt(ts1_sp.real[1:]**2+ts1_sp.imag[1:]**2)*np.exp([1j*theta[1:]])
#         print(nts1)
        if np.remainder(nts1,2)==0 :
            ts1_sp_rp[int(nts1/2.),n]=np.sqrt(2*ts1_sp.real[int(nts1/2.)]**2\
                                        +2*ts1_sp.imag[int(nts1/2.)]**2)\
                                *np.cos(theta[int(nts1/2.)])
        #power_rp=np.sqrt(ts1_sp_rp[:,n].real[:]**2+ts1_sp_rp[:,n].imag[:]**2)    
        #plt.figure(1)
        #plt.plot(freq[:nts1/2],power_rp[:nts1/2],'ro')
        #plt.plot(freq[:nts1/2],power[:nts1/2],'bx')
        #plt.show()
    
    #### perform inverse FFT on error_sp
    ts1_rp=np.zeros([nts1,nmember],dtype=np.float)
    for n in range(nmember) :
        ts1_rp[:,n]=np.fft.ifft(ts1_sp_rp[:,n])
        #plt.figure(1)
        #plt.plot(time,ts1_rp[:,n],'ro-')
        #plt.plot(time,ts1,'bx-')
        #plt.show()

    #### calculate correlation coeficient 
    corrcoef=np.zeros(nmember,dtype=np.float)
    for n in range(nmember) :
        corrcoef[n]=np.corrcoef([ts1,ts1_rp[:,n]])[0,1]

    #### calculate the 95% of noise generated correlation 
    nbin=nmember
    counts, bins = np.histogram(corrcoef, bins=nbin, range=(-1, 1))
    for i in range(nbin):
        totcount=np.sum(counts[:i])
        if totcount > nmember*conf_interv/100:
            conf_corrcoef=bins[i]
            break
           
    #plt.figure()
    #max_bin=np.max(corrcoef)
    #min_bin=np.min(corrcoef)
    #dbin=(max_bin-min_bin)/nbin
    #bin=min_bin+np.arange(nbin)*dbin
    #plt.hist(corrcoef,bins=bin)
    #plt.show()
   
    return conf_corrcoef


    
def stT_test_given(var1_mean,var2_mean,var1_std,var2_std,nvar1,nvar2,pcrit=0.975,type=1):
    """
    The function is designed for the student T hypothesis testing 
    with the indicated estimater value (var_mean), estimater standard deviation (var_std)
    and number of observation to derive the estimater value
    types
    1. Equal or Unequal sample sizes, unequal variances (Welch's t test)
        t  = (mean(var1)-mean(var2))/S
        S  = sqrt( variance1/n1 + variance2/n2 )
        df = ( variance1/n1 + variance2/n2 )^2 / ((variance1/n1)^2/(n1-1) + (variance2/n2)^2/(n2-1) )
    steps
    1) calculate the t value (tval)
    2) calculate the t value (tcrit) based on the p critical (pcrit) value set
    3) perform the hypothesis test
        |tval| > tcrit  => reject H0
        |tval| < tcrit  => accept H0
    """


    #### calculate t value
    if type == 1 :
        S=np.sqrt(var1_std**2/nvar1+var2_std**2/nvar2)
        df=S**2**2/(  (var1_std**2/nvar1)**2/(nvar1-1)  +  (var2_std**2/nvar2)**2/(nvar2-1)  )
        tval=(var1_mean-var2_mean)/S
        #print (var1_mean-var2_mean), S 

    #### calcualte t critical value based on p critical value
    tcrit=t.ppf(pcrit,df)

    #### test the null hypothesis
    if np.abs(tval) > np.abs(tcrit) :
       stTtest=1
       alpha=2.*(1.-pcrit)
       errorprob=alpha           # probability of type 1 error
       output='REJECT null hypothesis (two mean are the same), [tval,tcritical]=[%0.3f,%0.3f]'%(tval,tcrit)
    elif np.abs(tval) < np.abs(tcrit) :
       stTtest=0
       tcrit1=np.abs(tcrit)-np.abs(tval)
       tcrit2=np.abs(tcrit)+np.abs(tval)
       power1=1.-t.cdf(tcrit1,df)
       power2=1.-t.cdf(tcrit2,df)
       beta=1.-power1-power2
       errorprob=beta            # probability of type 2 error
       output='ACCEPT null hypothesis (two mean are the same), [tval,tcritical]=[%0.3f,%0.3f]'%(tval,tcrit)

    return stTtest,output,errorprob 



def stT_test(var1,var2,pcrit=0.975,type=1):
    """
    The function is designed for the student T hypothesis testing 
    with the indicated monthly error from the time series 
    types
    1. Equal or Unequal sample sizes, unequal variances (Welch's t test)
        t  = (mean(var1)-mean(var2))/S
        S  = sqrt( variance1/n1 + variance2/n2 )
        df = ( variance1/n1 + variance2/n2 )^2 / ((variance1/n1)^2/(n1-1) + (variance2/n2)^2/(n2-1) )
    steps
    1) calculate the t value (tval)
    2) calculate the t value (tcrit) based on the p critical (pcrit) value set
    3) perform the hypothesis test
        |tval| > tcrit  => reject H0
        |tval| < tcrit  => accept H0
    """


    nvar1=len(var1)
    nvar2=len(var2)

    #### calculate t value
    if type == 1 :
        S=np.sqrt(np.var(var1)/nvar1+np.var(var2)/nvar2)
        df=S**2**2/(  (np.var(var1)/nvar1)**2/(nvar1-1)  +  (np.var(var2)/nvar2)**2/(nvar2-1)  )
        tval=(np.mean(var1)-np.mean(var2))/S

    #### calcualte t critical value based on p critical value
    tcrit=t.ppf(pcrit,df)

    #### test the null hypothesis
    if np.abs(tval) > np.abs(tcrit) :
       stTtest=1
       alpha=2.*(1.-pcrit)
       errorprob=alpha           # probability of type 1 error
       output='REJECT null hypothesis (two mean are the same), [tval,tcritical]=[%0.3f,%0.3f]'%(tval,tcrit)
    elif np.abs(tval) < np.abs(tcrit) :
       stTtest=0
       tcrit1=np.abs(tcrit)-np.abs(tval)
       tcrit2=np.abs(tcrit)+np.abs(tval)
       power1=1.-t.cdf(tcrit1,df)
       power2=1.-t.cdf(tcrit2,df)
       beta=1.-power1-power2
       errorprob=beta            # probability of type 2 error
       output='ACCEPT null hypothesis (two mean are the same), [tval,tcritical]=[%0.3f,%0.3f]'%(tval,tcrit)

    return stTtest,output,errorprob 
    

    
def rmse(X1,X2,T1,T2) :
    """
    The function calculates the root mean square error between two time series     
    Input: (make sure the time axis of the two ts having the same axis)
      X1: time series array 1
      T1: time axis 1
      X2: time series array 2
      T2: time axis 2
    
    Output: 
      rmse:  root mean square error (rms value of the differences between two ts)
      nrmse: normalized root mean square error (rmse divided by the max value within the two ts)
      amp:   max value within the two ts
 
    Module included:
    check_match (module)
    numpy 
    sys 
    """
    #print '========='
#     flag=np.float(checkmatch_time(tim1=T1,tim2=T2)['flag'])
#     if flag == 1.:
#        sys.exit('ERROR: TIME ARRAY NOT MATCHING')
    #diff=X1[~np.isnan(X1)]-X2[~np.isnan(X2)]
    diff=X1-X2
    try:
        ddiff=diff[~np.isnan(diff)]
        AMP=np.max(np.concatenate((X1[~np.isnan(X1)],X2[~np.isnan(X2)])))-np.min(np.concatenate((X1[~np.isnan(X1)],X2[~np.isnan(X2)])))
        RMSE=np.sqrt(np.average(ddiff**2.))   
        NRMSE=RMSE/AMP
        #print 'RMSE: ',RMSE
        #print 'AMP: ',AMP
        #print 'NRMSE: ',NRMSE
    except ValueError:
        ddiff=np.float('nan')
        AMP=np.float('nan')
        RMSE=np.float('nan')  
        NRMSE=np.float('nan')

    return{'rmse': RMSE, 'nrmse': NRMSE, 'amp': AMP}


def variance_reduction(X1,X2) :
    """
    The function calculates the variance reduction of the two input time series. Defined by Tamisiea et al. 2010
    (the two time series is not required to be the same length)    
    IMPORTANT!!!!!!!   read before use!!!!!!
        idealy is using ts from obs as X1 and ts from obs minus the ts from derived SLF as X2
    Input: 
      X1: time series array 1
      X2: time series array 2
    
    Output: 
      var:  output the two variance from the two time series respectively. 
      var_red:  (var1-var2)/var1 * 100 (%)    
    Module included:
    numpy 
    sys 
    """
    var1=np.var(X1)
    var2=np.var(X2)
    var=[var1,var2]
    var_red=(var1-var2)/var1*100.
    
    return{'variance':var,'variance reduction':var_red}
    



#===================================================================================
# function below are still under testing 
    
def cross_cov(ts1,ts2):
    """
    The cross covariance is defined as 
        gamma_xy(tau)= E[(X[t]-Xmean) * (Y[t+tau]-Ymean)]
                     = sum_t{ (X[t]-Xmean)*(Y[t+tau]-Ymean) } / {size(X)}
                     = E[ X[t] * Y[t+tau]] - Xmean * Ymean
                     = sum_t{ X[t]*Y[t+tau] } / {size(X)} - Xmean * Ymean
        => sum_t{ X[t]*Y[t+tau] } = F(tau) = np.correlate (X,Y,mode='same') 
    """
    #ts1=ts1-np.average(ts1)
    #ts2=ts2-np.average(ts2)
    npts=len(ts1)
    if np.remainder(npts,2)==0:
        fir=np.arange(npts/2+1)+npts/2
        sec=npts-1-np.arange(npts/2-1)
    else:
        fir=np.arange(npts/2)+npts/2+1
        sec=npts-np.arange(npts/2+1)
    leng=np.concatenate((fir,sec),axis=0)
    crosscov=np.correlate(ts1,ts2,mode='same')/(leng-1)-np.average(ts1)*np.average(ts2)

#    npts=len(ts1)
#    fir=np.arange(npts)+1
#    sec=npts-1-np.arange(npts-1)
#    leng=np.concatenate((fir,sec),axis=0)
#    crosscov=np.correlate(ts1,ts2,mode='full')/(leng-1)-np.average(ts1)*np.average(ts2)

    #print np.average(ts1)
    #print np.average(ts2)
    #sys.exit('')
    return crosscov


def cross_corr(ts1,ts2):
    """
    The cross correlation is defined as 
        gamma_xy(tau)= E[(X[t]-Xmean) * (Y[t+tau]-Ymean)]
                     = sum_t{ (X[t]-Xmean)*(Y[t+tau]-Ymean) } / {size(X)}
                     = E[ X[t] * Y[t+tau]] - Xmean * Ymean
                     = sum_t{ X[t]*Y[t+tau] } / {size(X)} - Xmean * Ymean
        => sum_t{ X[t]*Y[t+tau] } = F(tau) = np.correlate (X,Y,mode='same')
    """
    #ts1=ts1-np.average(ts1)
    #ts2=ts2-np.average(ts2)
    npts=len(ts1)
    if np.remainder(npts,2)==0:
        fir=np.arange(npts/2+1)+npts/2
        sec=npts-1-np.arange(npts/2-1)
    else:
        fir=np.arange(npts/2)+npts/2+1
        sec=npts-np.arange(npts/2+1)
    leng=np.concatenate((fir,sec),axis=0)
    crosscorr=(np.correlate(ts1,ts2,mode='same')/leng-np.average(ts1)*np.average(ts2))/np.std(ts1)/np.std(ts2) # normalized cross-correlation
    #crosscorr=np.correlate(ts1-np.average(ts1),ts2-np.average(ts2),mode='same')
    

    return crosscorr


def cross_spec(ts1,ts2):
    """
    use the function cross_cov to determine the cross covariance 
        and then perform the Fourier transform to get the cross_spectrum
        output of coefficents, powers(ss), frequency is provided.
    """
    crosscov=cross_cov(ts1,ts2)
    #print crosscov
    nlag=crosscov.shape[0]
    
    crosscorr=cross_corr(ts1,ts2)
    #print crosscorr
    nlag=crosscorr.shape[0]

    # return crosscov only
    freq=np.fft.fftfreq(nlag)
    crosssp=np.fft.fft(crosscov)
    #print crosssp.real
    ss=np.abs(crosssp)**2                                                  # the other half repeat in opposite frequency
    phase=np.arctan(crosssp.imag[:]/crosssp.real[:])
    if np.remainder(nlag,2)==0:
        #ss[nlag/2]=(crosssp.real[nlag/2]**2+crosssp.imag[nlag/2]**2)      # there is no the other half at even number sampling 
        outsp=crosssp[:nlag/2+1]
        outpsd=ss[:nlag/2+1]
        outfreq=np.abs(freq[:nlag/2+1])
        outphase=phase[:nlag/2+1]
    else:
        outsp=crosssp[:(nlag-1)/2+1]
        outpsd=ss[:(nlag-1)/2+1]
        outfreq=np.abs(freq[:(nlag-1)/2+1])
        outphase=phase[:(nlag-1)/2+1]
        #print outfreq.shape,freq.shape
        #print outpsd.shape,crosssp.shape
    #sys.exit('')


    return outsp,outpsd,outfreq,outphase


def psd(ts1):
    #ts1=ts1-np.average(ts1)
    npts=ts1.shape[0]
    freq=np.fft.fftfreq(npts)
    sp=np.fft.fft(ts1)
    ss=np.abs(sp)**2                                # the other half repeat in opposite frequency
    #print sp
    #print np.abs(sp)
    #print np.abs(sp)**2
    #print np.abs(sp)**2/npts
    #print ss
    if np.remainder(npts,2)==0:
        #ss[npts/2]=np.abs(sp)**2/npts               # there is no the other half at even number sampling 
        outsp=sp[:npts/2+1]
        outpsd=ss[:npts/2+1]
        outfreq=np.abs(freq[:npts/2+1])
    else:   
        outsp=sp[:(npts-1)/2+1]
        outpsd=ss[:(npts-1)/2+1]
        outfreq=np.abs(freq[:(npts-1)/2+1])
    return outsp,outpsd,outfreq


def spectrum_coef(ts1,ts2):
    """
    use the function cross_spec to detemine the cross-spectrum value 
        and then calculate the gain function 
        Gxy=Sxy/Sxx
        ==============
        Gxy is the gain factor (function of frequency, or phase lag)
        can be thought of as the regression coef. of y(ts2) to x(ts1)
        Sxy is the cross-spectrum of y and x
        Sxx is the cross-spectrum of x and to itself.
        Gammaxy=Sxy^2/(Sxx*Syy)
        ==============
        Gammaxy is the coherence square
    """
    crosssp_xy,ss_xy,ff,phase_xy=cross_spec(ts1,ts2)
    crosssp_xx,ss_xx,freq,phase_xx=cross_spec(ts1,ts1)
    crosssp_yy,ss_yy,freq,phase_yy=cross_spec(ts2,ts2)
    #sp_xx,psd_xx,ffxx=psd(ts1)
    #sp_yy,psd_yy,ffyy=psd(ts2)
    #psd_xx_test = signal.periodogram(ts1)
    #print psd_xx
    #print crosssp_xx.real*len(ts1)
    #print np.abs(sp_xx)**2
    print(np.abs(crosssp_xx)**2/(psd_xx*psd_xx))
    #sys.exit('')
    gfactor=ss_xy/ss_xx
    gfactor=gfactor.real
    cohersq=ss_xy**2/(ss_xx*ss_yy)
    cohersq=np.abs(cohersq.real)
    plt.subplot(411)
    plt.semilogx(1/ff, ss_xy)
    plt.subplot(412)
    plt.semilogx(1/ff, phase_xy)
    plt.subplot(413)
    plt.semilogx(1/ff, gfactor)
    plt.subplot(414)
    plt.semilogx(1/ff, cohersq)
    plt.show()

    return gfactor,cohersq,ff


