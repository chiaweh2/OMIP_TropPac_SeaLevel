import xarray as xr
import cftime
import numpy as np


"""
The following function are related to extracting the 
ENSO related composite

"""



def enso_event_recog(da_oni,enso_event='ElNino',\
              enso_crit_lower=0.5,enso_crit_max=None,\
              enso_crit_min=None,enso_cont_mon=5):
    """
    Create xr.Dataset which include the picked ENSO event info
    
    input :
    
        da_oni(xr.DataArray) - Oceanic Nino index time series 
    
    Parameters :
    
        enso_event(string) - There are two options 'ElNino' or 'LaNina'.
    
        enso_crit_lower(float) - The lowest ONI value to defined ALL picked event
        default to 0.5 based on the Climate Prediction Center at NOAA.
        
        enso_crit_max(float) - The maxmium ONI value in the a range which is used to picked 
        the desired category of event based on ONI value. Default = None, which 
        means ALL events are included. 
        
        enso_crit_min(float) - The minimium ONI value in the a range which is used to picked 
        the desired category of event based on ONI value. Default = None, which 
        means ALL events are included.
        
        enso_cont_mon(int) - The minimum continuous number of months to be defined as a event 
        period.
        
    Returns:
        
        ds_enso(xr.Dataset) - New dataset includes variables like
        time series :
            "event_length" for the total number of month included in an event,
            "event_number" to index an event,
            "event_enso" shows event which satisfied the enso_crit_lower, enso_crit_max, and enso_crit_min
            "all_enso" shows ALL event satisfied the enso_crit_lower
            "ori_oni" is the original da_oni
        values :
            "crit_max_oni", "crit_min_oni", and "enso_type" to store the setting
            
    """


    if enso_event is 'LaNina':
        da_enso = -da_oni
    elif enso_event is 'ElNino':
        da_enso = da_oni
    else:
        print('please input correct enso type ("LaNina" or "ElNino" )')
        print(enso_event)

    #### identify monthly El Nino event
    da_enso = da_enso.where(da_enso > enso_crit_lower, other=np.nan) 

    #### counting total El Nino events
    da_enso_all=da_enso.copy()+np.nan
    da_enso_crit=da_enso.copy()+np.nan
    da_enso_eventnum=da_enso.copy()+np.nan
    da_enso_eventlen=da_enso.copy()+np.nan
    enso_event_count=0
    event_length=0
    temp_ind=[]
    for kk in range(len(da_enso)):
        if da_enso.values[kk] > 0. :
            event_length+=1
            temp_ind.append(kk)
        else:
            if event_length>=enso_cont_mon: 
                da_enso_all[temp_ind]=da_enso[temp_ind]
                if enso_crit_max and enso_crit_min:
                    da_temp=da_enso[temp_ind]#.rolling(dim={"time":3},min_periods=3,center=True).mean()
                    if da_temp.max() <= enso_crit_max and da_temp.max() >= enso_crit_min :
                        enso_event_count+=1
                        da_enso_crit[temp_ind]=da_enso[temp_ind]
                        da_enso_eventnum[temp_ind]=enso_event_count
                        da_enso_eventlen[temp_ind]=event_length
                elif enso_crit_max is None and enso_crit_min is None:
                    enso_event_count+=1
                    da_enso_crit[temp_ind]=da_enso[temp_ind]
                    da_enso_eventnum[temp_ind]=enso_event_count
                    da_enso_eventlen[temp_ind]=event_length
                else:
                    sys.exit('please put both min and max El Nino Criterias or else put both as None')

            temp_ind=[]
            event_length=0

    if enso_event is 'LaNina':
        da_enso_crit = -da_enso_crit
        da_enso_all = -da_enso_all

    ds_enso = xr.Dataset()
    ds_enso['event_length'] = da_enso_eventlen
    ds_enso['event_number'] = da_enso_eventnum
    ds_enso['event_enso'] = da_enso_crit
    ds_enso['all_enso'] = da_enso_all
    ds_enso['event_enso_mask'] = da_enso_crit.where(da_enso_crit.isnull(),other=1.)
    ds_enso['ori_oni'] = da_oni
    if enso_crit_max and enso_crit_min:
        ds_enso['crit_max_oni'] = enso_crit_max
        ds_enso['crit_min_oni'] = enso_crit_min

    if enso_event is 'LaNina':
        ds_enso['enso_type'] = 'La_Nina'
    elif enso_event is 'ElNino':
        ds_enso['enso_type'] = 'El_Nino'
        
    return ds_enso



def enso_comp(ds_enso,da_var,premon=0,postmon=0):
    """
    Create xr.Dataset that contain the da_var seperated to individual time 
    serie which is based on each ENSO period. This is designed to create the 
    composite of ENSO event for different variables.
    
    input :
    
        ds_enso(xr.Dataset) - the output from function enso_event_recog()
        
        da_var(xr.DataArray) - the desided variable one wants to create a 
        composite dataset which sperate each event to individual variables 
        (xr.DataArray) in a xr.Dataset
        
    Parameters :
    
        premon(int) - the number of month one want to count into the 
        event before the month when the ONI threshold is reached. Default = 0
        which means each event start when the ONI threshold is reached
        
        postmon(int) - the number of month one want to count into the 
        event after the month when the ONI value start to become lower 
        than the threshold. Default = 0 which means each event end 
        at the month when ONI value is lower than the threshold.     
        
    Returns:
        
        ds_var_enso_comp(xr.Dataset) - New dataset includes variables like
        time series :
            "event%i" is the time series for each event,
            "cftime" is the time coordinate start from 0000-01-01
            "time" is the time coordinate start from 2000-01-01            
        attributes :
            "begin-end year" is in each "event%i"(xr.DataArray) indicating
            the real start year and end year of the event.
            
    """
    
    ds_var_enso_comp = xr.Dataset()
    ds_var_enso_comp_list = []
    for index in np.unique(ds_enso.event_number.where(ds_enso.event_number.notnull(),drop=True)):
        #print(index)
#         da_enso_single = da_var.where(ds_enso.event_number==index,drop=True)

        # find event begin/end year
        begin_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[0]
        end_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[-1]

        timeindex = np.arange(len(ds_enso.event_enso))[(ds_enso.event_number==index).values]
        minmon = timeindex.min()-premon
        if minmon <= 0:
            minmon = 0
        maxmon = timeindex.max()+postmon
        if maxmon >= (len(ds_enso.event_enso)-1):
            maxmon = len(ds_enso.event_enso)-1
        timeindex_new = np.arange(minmon,maxmon+1,dtype=int)

        
        da_enso_single = da_var.isel(time=timeindex_new).copy()

        da_enso_single['year'] = da_enso_single['time.year']-da_enso_single['time.year'].min()
        da_enso_single['month'] = da_enso_single['time.month']
        cftime_list = [cftime.datetime(da_enso_single['year'][ii].values,da_enso_single['month'][ii].values,1) \
                       for ii in range(len(da_enso_single))]
        time_list = ["%0.4i-%0.2i-01"%(da_enso_single['year'][ii].values+2000,da_enso_single['month'][ii].values) \
                       for ii in range(len(da_enso_single))]
        da_enso_single['year'].values = xr.CFTimeIndex(cftime_list).values
        #da_enso_single['month'].values = da_enso_single['time'].values
        da_enso_single['time'] = np.array(time_list,dtype=np.datetime64)
        da_enso_single = da_enso_single.rename({'year':'cftime'})
        da_enso_single.name = 'event%i'%(index)
        da_enso_single.attrs = {'begin-end year':'%0.4i-%0.4i'%(begin_year,end_year)}
        ds_var_enso_comp_list.append(da_enso_single)
        
    ds_var_enso_comp=xr.merge(ds_var_enso_comp_list)
        
    return ds_var_enso_comp



def enso_comp_maxval(ds_enso,da_var,premon=12,postmon=12):
    """
    Create xr.Dataset that contain the da_var seperated to individual time 
    serie which is based on the period defined by the maxmium value of ONI in each 
    ENSO event. The period includes the number of month (premon) before the maximum 
    ONI value till the number of month (postmon) after the maximum ONI value. This 
    is designed to create the composite of ENSO event for different variables.
    
    input :
    
        ds_enso(xr.Dataset) - the output from function enso_event_recog()
        
        da_var(xr.DataArray) - the desided variable one wants to create a 
        composite dataset which sperate each event to individual variables 
        (xr.DataArray) in a xr.Dataset 
        
    Parameters :
    
        premon(int) - each event period starts from a number of month (premon) 
        before the maximum ONI value is reached. Default = 12 which means each event 
        start 12 months before the ONI reach the maximum. 
        
        postmon(int) - each event period ends at number of month (postmon) 
        after the maximum ONI value is reached. Default = 12 which means each event 
        ends at 12 months after the ONI reach the maximum.      
        
    Returns:
        
        ds_var_enso_comp(xr.Dataset) - New dataset includes variables like
        time series :
            "event%i" is the time series for each event,
            "cftime" is the time coordinate start from 0000-01-01
            "time" is the time coordinate start from 2000-01-01            
        attributes :
            "begin-end year" is in each "event%i"(xr.DataArray) indicating
            the real start year and end year of the event.
            
    """    
    ds_var_enso_comp = xr.Dataset()
    ds_var_enso_comp_list = []
    
    for index in np.unique(ds_enso.event_number.where(ds_enso.event_number.notnull(),drop=True)):
        
        # find event begin/end year
        begin_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[0]
        end_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[-1]
      
        
        timeindex = np.arange(len(ds_enso.event_enso))[(ds_enso.event_number==index).values]
        max_ind = timeindex[np.argmax(np.abs(ds_enso.event_enso.isel(time=timeindex)))]
        
        minmon = max_ind-premon
        if minmon <= 0:
            minmon = 0
        maxmon = max_ind+postmon
        if maxmon >= (len(ds_enso.event_enso)-1):
            maxmon = len(ds_enso.event_enso)-1
        timeindex_new = np.arange(minmon,maxmon+1,dtype=int)

        
        da_enso_single = da_var.isel(time=timeindex_new).copy()

        da_enso_single['year'] = da_enso_single['time.year']-da_enso_single['time.year'].min()
        da_enso_single['month'] = da_enso_single['time.month']
        cftime_list = [cftime.datetime(da_enso_single['year'][ii].values,da_enso_single['month'][ii].values,1) \
                       for ii in range(len(da_enso_single))]
        time_list = ["%0.4i-%0.2i-01"%(da_enso_single['year'][ii].values+2000,da_enso_single['month'][ii].values) \
                       for ii in range(len(da_enso_single))]
        da_enso_single['year'].values = xr.CFTimeIndex(cftime_list).values
        #da_enso_single['month'].values = da_enso_single['time'].values
        da_enso_single['time'] = np.array(time_list,dtype=np.datetime64)
        da_enso_single = da_enso_single.rename({'year':'cftime'})
        da_enso_single.name = 'event%i'%(index)
        da_enso_single.attrs = {'begin-end year':'%0.4i-%0.4i'%(begin_year,end_year)}
        ds_var_enso_comp_list.append(da_enso_single)
        
    ds_var_enso_comp=xr.merge(ds_var_enso_comp_list)
        
    return ds_var_enso_comp



def enso_comp_map_mean_maxval(ds_enso,da_var,premon=12,postmon=12,exclude_event_num=None):
    """
    Create xr.Dataset that contain the composite mean of da_var (gridded) 
    which is based on the period defined by the maxmium value of ONI in each 
    ENSO event. The period includes the number of month (premon) before the maximum 
    ONI value till the number of month (postmon) after the maximum ONI value. This 
    is designed to create the composite mean of ENSO event for different variables.
    
    input :
    
        ds_enso(xr.Dataset) - the output from function enso_event_recog()
        
        da_var(xr.DataArray) - the desided variable one wants to create a 
        composite dataset which sperate each event to individual variables 
        (xr.DataArray) in a xr.Dataset 
        
    Parameters :
    
        premon(int) - each event period starts from a number of month (premon) 
        before the maximum ONI value is reached. Default = 12 which means each event 
        start 12 months before the ONI reach the maximum. 
        
        postmon(int) - each event period ends at number of month (postmon) 
        after the maximum ONI value is reached. Default = 12 which means each event 
        ends at 12 months after the ONI reach the maximum.      
        
    Returns:
        
        ds_var_enso_comp(xr.Dataset) - New dataset includes variables like
        time series :
            "event%i" is the time series for each event,
            "cftime" is the time coordinate start from 0000-01-01
            "time" is the time coordinate start from 2000-01-01            
        attributes :
            "begin-end year" is in each "event%i"(xr.DataArray) indicating
            the real start year and end year of the event.
            
    """    
    #ds_var_enso_comp = xr.Dataset()
    #ds_var_enso_comp_list = []
    
    count = 1
    for index in np.unique(ds_enso.event_number.where(ds_enso.event_number.notnull(),drop=True)):
        
        # find event begin/end year
        begin_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[0]
        end_year = ds_enso.where(ds_enso.event_number==index,drop=True)['time.year'].values[-1]
        
        timeindex = np.arange(len(ds_enso.event_enso))[(ds_enso.event_number==index).values]
        max_ind = timeindex[np.argmax(np.abs(ds_enso.event_enso.isel(time=timeindex)))]
        
        minmon = max_ind-premon
        if minmon <= 0:
            minmon = 0
        maxmon = max_ind+postmon
        if maxmon >= (len(ds_enso.event_enso)-1):
            maxmon = len(ds_enso.event_enso)-1
        timeindex_new = np.arange(minmon,maxmon+1,dtype=int)

#         print(begin_year,end_year)
#         print(timeindex_new)
        da_enso_single = da_var.isel(time=timeindex_new).copy()

        da_enso_single['year'] = da_enso_single['time.year']-da_enso_single['time.year'].min()
        da_enso_single['month'] = da_enso_single['time.month']
        cftime_list = [cftime.datetime(da_enso_single['year'][ii].values,da_enso_single['month'][ii].values,1) \
                       for ii in range(len(da_enso_single))]
        time_list = ["%0.4i-%0.2i-01"%(da_enso_single['year'][ii].values+2000,da_enso_single['month'][ii].values) \
                       for ii in range(len(da_enso_single))]
        da_enso_single['year'].values = xr.CFTimeIndex(cftime_list).values
        #da_enso_single['month'].values = da_enso_single['time'].values
        da_enso_single['time'] = np.array(time_list,dtype=np.datetime64)
        da_enso_single = da_enso_single.rename({'year':'cftime'})
        #da_enso_single.name = 'event%i'%(index)
        #da_enso_single.attrs = {'begin-end year':'%0.4i-%0.4i'%(begin_year,end_year)}
        #ds_var_enso_comp_list.append(da_enso_single)
        
        if index not in exclude_event_num:
            if count == 1 :
                da_enso_comp_mean = da_enso_single.copy()
            else:
                da_enso_comp_mean = da_enso_comp_mean+da_enso_single
            count += 1
            
    da_enso_comp_mean = da_enso_comp_mean/index
    
    #ds_var_enso_comp=xr.merge(ds_var_enso_comp_list)
        
    return da_enso_comp_mean



def enso_comp_concat_index(ds_enso,premon=0,postmon=0,exclude_event_num=None):
    """
    Create a np.array of index indicating all ENSO periods. This is designed 
    to create the composite of ENSO event for different variables.
    
    input :
    
        ds_enso(xr.Dataset) - the output from function enso_event_recog()
        
    Parameters :
    
        premon(int) - the number of month one want to count into the 
        event before the month when the ONI threshold is reached. Default = 0
        which means each event start when the ONI threshold is reached
        
        postmon(int) - the number of month one want to count into the 
        event after the month when the ONI value start to become lower 
        than the threshold. Default = 0 which means each event end 
        at the month when ONI value is lower than the threshold.
        
        exclude_event_num(list of int) - include the event number one want to 
        exclude when calculating the mean. Default = None means no events are 
        excluded.
        
    Returns:
        
        index_concat(np.array) - np.array of index indicating all ENSO periods
            
    """
    
    index_list = []
    for index in np.unique(ds_enso.event_number.where(ds_enso.event_number.notnull(),drop=True)):
        if index not in exclude_event_num:
            timeindex = np.arange(len(ds_enso.event_enso))[(ds_enso.event_number==index).values]
            minmon = timeindex.min()-premon
            if minmon <= 0:
                minmon = 0
            maxmon = timeindex.max()+postmon
            if maxmon >= (len(ds_enso.event_enso)-1):
                maxmon = len(ds_enso.event_enso)-1
            timeindex_new = np.arange(minmon,maxmon+1,dtype=int)

            index_list.append(timeindex_new)
        
    index_concat=np.concatenate(index_list)
        
    return index_concat


def enso_comp_mean(ds_var_enso_comp,eventname='event',exclude_event_num=None,skipna=True):
    """
    Added a new xr.DataArray in the ds_var_enso_comp (xr.Dataset) which represents the 
    mean of the picked events. It also remove the excluded event in the updated ds_var_enso_comp
    
    input :
    
        ds_var_enso_comp(xr.Dataset) - the output from function enso_comp() 
        or enso_comp_maxval()
        
    Parameters :
    
        eventname(string) - name of each ENSO event. Default = 'event'. If 
        enso_comp() or enso_comp_maxval() function are not modified to update 
        the name of each event this keyword argument should not be changed. 
        
        exclude_event_num(list of int) - include the event number one want to 
        exclude when calculating the mean. Default = None means no events are 
        excluded.
        
        skipna - when calculating the mean (using xr.DataArray.mean()) if one want
        to skip NaN or not skip the NaN. Default = True skip the NaN
        
    Returns:
        
        ds_var_enso_comp(xr.Dataset) - updated input dataset with the xr.DataArray 
        showing the mean time series.
    
    """
         
    if exclude_event_num is None:
        count = 0
        for name,da in ds_var_enso_comp.data_vars.items():
            if eventname in name:
                count += 1
                da_temp = ds_var_enso_comp[name].copy()
                if count == 1 :
                    da_ts = da_temp.copy()
                else:
                    da_ts = xr.concat([da_ts,da_temp],dim=eventname)
        ds_var_enso_comp["composite_mean"] = da_ts.mean(dim=eventname,skipna=skipna)
        
    else:
        exclude_event=["%s%i"%(eventname,num) for num in exclude_event_num]
        count = 0
        for name,da in ds_var_enso_comp.data_vars.items():
            if eventname in name and name not in exclude_event:
                count += 1 
                da_temp = ds_var_enso_comp[name].copy()
                if count == 1 :
                    da_ts = da_temp
                else:
                    da_ts = xr.concat([da_ts,da_temp],dim=eventname)
        ds_var_enso_comp["composite_mean_exc"] = da_ts.mean(dim=eventname,skipna=skipna)
        ds_var_enso_comp["composite_mean_exc"].attrs['exclude'] = exclude_event
    
    return ds_var_enso_comp




######################################################
# streamline the enso map composite processing
import matplotlib.pyplot as plt

def elnino_composite(da_oni,syear,fyear,premon=13,postmon=15,period_type='maxval',exclude=[]):
    # use the function iteratively to remove the unwantted event 
    # with the option exclude. If exclude is not specified all 
    # event is used to calculate the mean event

    # crop data (time)
    da_oni = da_oni.where((da_oni['time.year'] >= syear)&\
                          (da_oni['time.year'] <= fyear)
                          ,drop=True)

    # using oni to create enso event DataSet & index
    ds_enso = enso_event_recog(da_oni,
                               enso_event='ElNino',
                               enso_crit_lower=0.5,
                               enso_crit_max=None,
                               enso_crit_min=None,
                               enso_cont_mon=5)

    # find total index 
    event_index = np.unique(ds_enso.event_number.where(ds_enso.event_number.notnull()
                                                       ,drop=True))
    
    # ONI composite period
    #   maxval is prefered and default option
    if period_type in ['maxval']:
        #  composite period is #premon before the maxoni in the identified El Nino
        #  and #postmon after the maxoni in the identified El Nino
        #  keep the picked period with same length
        ds_oni_comp = enso_comp_maxval(ds_enso,
                                      da_oni,
                                      premon=premon,
                                      postmon=postmon)
    elif period_type in ['ensoperiod']:
         #  composite period is #premon before the elnino period initial month 
         #  and #postmon after the elnino period final month 
         #  keep the entire elnino period but may have very different El Nino event length
        ds_oni_comp = enso_comp(ds_enso,
                               da_oni,
                               premon=premon,
                               postmon=postmon)


    # calculate mean oni excluding event
    ds_oni_comp = enso_comp_mean(ds_oni_comp,exclude_event_num=exclude,skipna=False)
    
    # plotting code to quickly pick the     
    ds = ds_oni_comp.copy()
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_axes([0,0,1,0.5])
    exc = ["%s%i"%('event',num) for num in exclude]
    for name,da in ds.data_vars.items() : 
        if 'exc' in name:
            ds[name].plot(ax=ax,label=name,linewidth=2.0,color='k')
        elif 'event' in name and name not in exc:   
            ds[name].plot(ax=ax,label='%s_%s'%(name,ds_oni_comp[name].attrs['begin-end year']),linestyle='dashed')

    ax.set_title("El Nino Composite")
    ax.set_ylabel('ONI ($^\circ$C)',{'size':'15'})
    ax.tick_params(axis='y',labelsize=15) 
    ax.set_xticks(ds.time.values[::2])
    ax.set_xticklabels(["Year%0.1i-%0.2i"%(date.year,date.month) for date in ds[name].cftime.values[::2]])
    ax.set_xlabel('',{'size':'15'})
    ax.tick_params(axis='x',labelsize=15,rotation=70)
    ax.legend(frameon=False)
    
    
    
    return ds_oni_comp

        

def proc_elnino_composite_maps(da_oni,da_var,syear,fyear,premon=13,postmon=15,period_type='maxval',exclude=[]):
    
    # crop data (time)
    da_oni = da_oni.where((da_oni['time.year'] >= syear)&\
                          (da_oni['time.year'] <= fyear)
                          ,drop=True)

    # using oni to create enso event DataSet & index
    ds_enso = enso_event_recog(da_oni,
                               enso_event='ElNino',
                               enso_crit_lower=0.5,
                               enso_crit_max=None,
                               enso_crit_min=None,
                               enso_cont_mon=5)
    
    if period_type in ['maxval']:
        # calculate composite
        da_elnino_comp = enso_comp_map_mean_maxval(ds_enso,
                                                   da_var,
                                                   premon=premon,
                                                   postmon=postmon,
                                                   exclude_event_num=exclude)
        return da_elnino_comp
    else:
        print("Method for picking the El Nino composite not available")
        


   



