#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from enso_composite import enso_event_recog,enso_comp_maxval,enso_comp_mean


"""
The script generate the ONI index and ENSO composite plot
in the form of
figure 20 : time series of ONI index in all El Nino events and comparison with obs
in the paper.

input files
============
JRA55-do/CORE : tos, oni (exec io_gfdl_tos_oni.py first)

observation : Hist_ONI_cpc_noaa.txt (ONI index download from CPC NOAA)


function used
==================
create_ocean_mask.levitus98 : which generate the Pacific basin mask
spherical_area.cal_area     : generate area array based on the lon lat of data
enso_composite.enso_event_recog : picked out the ENSO events
enso_composite.enso_comp_maxval : picked out the event for ENSO composite
enso_composite.enso_comp_mean : calculate the ENSO composite mean

"""
# noi
path = './data/GFDL/JRA/regional_avg/'
file = 'JRA_tos_oni_ts.nc'
ds_oni = xr.open_dataset(path+file)
da_oni = ds_oni.oni

syear = 1958
fyear = 2007

premon=13
postmon=15

period_type='maxval'
exclude=[1]

# crop data (time)
da_oni = da_oni.where((da_oni['time.year'] >= syear)&\
                      (da_oni['time.year'] <= fyear)\
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
#         ds[name].plot(ax=ax,label='%s_%s'%(name,ds_oni_comp[name].attrs['begin-end year']),linestyle='dashed')
        ds[name].plot(ax=ax,label='%s'%(ds_oni_comp[name].attrs['begin-end year']),linestyle='dashed')

ax.set_title("El Nino Composite")
ax.set_ylabel('ONI ($^\circ$C)',{'size':'15'})
ax.tick_params(axis='y',labelsize=15)
ax.set_xticks(ds.time.values[::2])
ax.set_xticklabels(["Year%0.1i-%0.2i"%(date.year,date.month) for date in ds[name].cftime.values[::2]])
ax.set_xlabel('',{'size':'15'})
ax.tick_params(axis='x',labelsize=15,rotation=70)
ax.legend(frameon=False)

fig.savefig('./figure/figure20a.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)





# # Calculate the Oceanic Nino Index
# Nino3.4 index use the OMIP model output "tos" which is representing the sea surface temperaturesea (SST) usually measured over the ocean. Warm (red) and cold (blue) periods based on a threshold of +/- 0.5C for the Oceanic Nino Index (ONI) [3 month running mean of ERSST.v5 SST anomalies
# average over the Pacific Ocean tropic region in the Nino 3.4 region (5N-5S, 120-170W)], based on centered 30-year base periods updated every 5 years.
# (http://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)


import os
import cftime
import dask
import xarray as xr
import numpy as np
import nc_time_axis
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=1, processes=False)

from mem_track import used_memory
used_memory()

# # OMODEL file detail
#### possible input info from external text file
# constant setting
syear = 1948
fyear = 2017
tp_lat_region = [-30,30]     # extract model till latitude

Model_varname = ['tos']
Area_name = ['areacello']

Model_name = ['JRA','CORE']

# standard model (interpolated to this model)
Model_standard = 'JRA'
Variable_standard = 'tos'
modeldir = './data/GFDL/JRA/'
modelfile = 'JRA_tos.zarr'
path_standard = modeldir+modelfile

# inputs
modelin = {}
path = {}
model = Model_name[0]
modeldir = './data/GFDL/JRA/'
modelfile = [['JRA_tos.zarr']]
path[model]=[modeldir,modelfile]

model = Model_name[1]
modeldir = './data/GFDL/CORE/'
modelfile = [['CORE_tos.zarr']]
path[model]=[modeldir,modelfile]



for nmodel,model in enumerate(Model_name):
    modeldir = path[model][0]
    modelfile = path[model][1]
    multivar = []
    for file in modelfile :
        if len(file) == 1 :
            multivar.append([os.path.join(modeldir,file[0])])
        elif len(file) > 1 :
            multifile = []
            for ff in file :
                multifile.append(os.path.join(modeldir,ff))
            multivar.append(multifile)
    modelin[model] = multivar

# # Read OMODEL dataset
#
# read in as dask array to avoid memory overload

import warnings
warnings.simplefilter("ignore")

# initialization of dict and list  (!!!!!!!! remove all previous read model info if exec !!!!!!!!!!)
nmodel = len(Model_name)
nvar = len(Model_varname)

ds_model_mlist = {}
mean_mlist = {}
season_mlist = {}

#### models
import sys
for nmodel,model in enumerate(Model_name):
    ds_model_list = {}
    mean_list = {}
    season_list = {}
    for nvar,var in enumerate(Model_varname):
        print('read %s %s'%(model,var))

        # read input data
        #-- single file
        if len(modelin[model][nvar]) == 1 :
            ds_model = xr.open_zarr(modelin[model][nvar][0])
        #-- multi-file merge (same variable)
        elif len(modelin[model][nvar]) > 1 :
            for nf,file in enumerate(modelin[model][nvar]):
                ds_model_sub = xr.open_zarr(file)
                if nf == 0 :
                    ds_model = ds_model_sub
                else:
                    ds_model = xr.concat([ds_model,ds_model_sub],dim='time',data_vars='minimal')

        # crop data (time)
        da_model = ds_model[var]\
                          .where((ds_model['time.year'] >= syear)&\
                                 (ds_model['time.year'] <= fyear)\
                                 ,drop=True)
        da_model = da_model\
                          .where((ds_model.lat >= np.min(np.array(tp_lat_region)))& \
                                (ds_model.lat <= np.max(np.array(tp_lat_region)))\
                                ,drop=True)

        # store all model data
        ds_model_list[var] = da_model

        # calculate mean
        mean_list[var] = ds_model_list[var].mean(dim='time').compute()
        ds_model_list[var] = ds_model_list[var]-mean_list[var]

        # calculate seasonality
        season_list[var] = ds_model_list[var].groupby('time.month').mean(dim='time').compute()
        ds_model_list[var] = ds_model_list[var].groupby('time.month')-season_list[var]

    mean_mlist[model] = mean_list
    season_mlist[model] = season_list
    ds_model_mlist[model] = ds_model_list


# initialize dictionary  (exec this cell will remove all previous calculated values for all variables)
regional_var_mlist = {}

for nmodel,model in enumerate(Model_name):
    regional_var_mlist[model] = xr.Dataset()


# # Regional average of SST to derive ONI
# - regional average (170-120W, 5S-5N)


# Model
varname = 'tos'
lon_range_list = [[-170,-120]]    # Lon: -180-180
lat_range_list = [[-5,5]]        # Lat: -90-90

# #### output dir
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/regional_avg/'
# modelfile = 'JRA_%s_oni_ts.nc'%(varname)

varind = Model_varname.index(varname)


for nmodel,model in enumerate(Model_name):
    for nn in range(len(lon_range_list)):
        print('process',lon_range_list[nn])

        #### setting individual event year range
        lon_range  = lon_range_list[nn]
        lat_range  = lat_range_list[nn]

        # correct the lon range
        lon_range_mod = np.array(lon_range)
        lonmin = ds_model_mlist[model][varname].lon.min()
        ind1 = np.where(lon_range_mod>np.float(360.+lonmin))[0]
        lon_range_mod[ind1] = lon_range_mod[ind1]-360.         # change Lon range to -300-60 (might be different for different model)


        # read areacello
        da_area = xr.open_zarr(modelin[model][varind][0])[Area_name[varind]]
        da_area = da_area.where(\
                              (da_area.lon>=np.min(lon_range_mod))&\
                              (da_area.lon<=np.max(lon_range_mod))&\
                              (da_area.lat>=np.min(lat_range))&\
                              (da_area.lat<=np.max(lat_range))\
                               ).compute()

        # calculate the temporal mean of regional mean
        mean_var = mean_mlist[model][varname]
        regional_mean = (mean_var*da_area).sum(dim=['x','y'])/(da_area).sum()

        # calculate time varying regional mean
        regional_var_mlist[model]['oni']\
             = ((ds_model_mlist[model][varname]*da_area).sum(dim=['x','y'])
              /(da_area).sum()).compute()
        regional_var_mlist[model]['oni'] = regional_var_mlist[model]['oni']+regional_mean

        # calculate 3 month moving average
        regional_var_mlist[model]['oni']\
            = regional_var_mlist[model]['oni'].rolling(dim={"time":3},min_periods=3,center=True).mean()


        # removing 30 year mean for each 5 year period located at the center of the 30 year window
        moving_window=30                                   # years
        num_year_removemean=5                              # years
        da_oni_noclim=regional_var_mlist[model]['oni'].copy()
        da_moving_mean=np.zeros(len(da_oni_noclim))
        for ii in range(0,len(da_oni_noclim),num_year_removemean*12):
            if ii < moving_window/2*12 :
                da_moving_mean[ii:ii+5*12]=da_oni_noclim[:ii+15*12].mean().values
            elif ii > len(da_oni_noclim)-moving_window/2*12:
                da_moving_mean[ii:ii+5*12]=da_oni_noclim[-15*12+ii:].mean().values
            else:
                da_moving_mean[ii:ii+5*12]=da_oni_noclim[ii-15*12:ii+15*12].mean().values
        regional_var_mlist[model]['oni']=da_oni_noclim-da_moving_mean



# # plotting ONI (Model V.S. Observation from CPC)
import pandas as pd
import datetime
table1=pd.read_csv('/storage1/home1/chiaweih/Research/proj1_enso_temp/data/Hist_ONI_cpc_noaa.txt',sep='\s+',header=1)
nrow,ncol=table1.shape
date=[datetime.datetime(table1['YR'].values[i], table1['MON'].values[i],1) for i in range(nrow)]
da_oni_cpc=xr.DataArray(table1['ANOM'].values,coords={'time':date},dims='time')


import matplotlib.transforms as mtransforms
elnino_crit_lower=0.5
elnino_crit_max=None
elnino_crit_min=None
elnino_cont_mon=5
starttime=date[0].year
endtime=date[-1].year

#### identify monthly El Nino event
da_elnino=da_oni_cpc.copy()
da_elnino.values[da_elnino<elnino_crit_lower]+=np.nan

#### counting total El Nino events
da_elnino_crit=da_elnino.copy()+np.nan
elnino_event_count=0
event_length=0
temp_ind=[]
for kk in range(len(da_elnino)):
    if da_elnino.values[kk] > 0. :
        event_length+=1
        temp_ind.append(kk)
    else:
        if event_length>=elnino_cont_mon:
            if elnino_crit_max and elnino_crit_min:
                # 3 month mean for all available continuous El Nino months
                da_temp=da_elnino[temp_ind].rolling(dim={"time":3},min_periods=3,center=True).mean()
                ##print da_temp.max(),da_elnino[temp_ind].time.values[0]
                # any 3 month mean lower than the critical max value and larger than critical min value
                # will be catagorized accordingly
                if da_temp.max() <= elnino_crit_max and da_temp.max() >= elnino_crit_min :
                    elnino_event_count+=1
                    da_elnino_crit[temp_ind]=da_elnino[temp_ind]
            elif elnino_crit_max is None and  elnino_crit_min is None:
                elnino_event_count+=1
                da_elnino_crit[temp_ind]=da_elnino[temp_ind]
            else:
                sys.exit('please put both min and max El Nino Criterias or else put both as None')

        temp_ind=[]
        event_length=0

#### plotting
fig=plt.figure(1)
ax1=fig.add_axes([0,0,2,1])
ax1color='k'

modelname=['JRA','CORE']
model_legend_name = ['JRA55-do','CORE']
modelcolor=['C0','C1']

for nmodel,model in enumerate(modelname):
    regional_var_mlist[model]['oni'].plot(ax=ax1,
                                          label=model_legend_name[nmodel],
                                          linewidth=3.0,
                                          color=modelcolor[nmodel],
                                          alpha=0.9)

da_oni_cpc.plot(ax=ax1,label="CPC",color=ax1color,linestyle='--')


#### plot event
if elnino_crit_max and elnino_crit_min:
    elnino_low_line=np.zeros(len(regional_var_mlist['JRA']['oni']))+elnino_crit_min
    elnino_max_line=np.zeros(len(regional_var_mlist['JRA']['oni']))+elnino_crit_max
    ax1.plot(regional_var_mlist['JRA']['oni'].time.values,elnino_low_line,color=ax1color,linestyle='dashed',alpha=0.5)
    ax1.plot(regional_var_mlist['JRA']['oni'].time.values,elnino_max_line,color=ax1color,linestyle='dashed',alpha=0.5)
trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
# ax1.fill_between(regional_var_mlist['JRA']['oni'].time.values,0,1\
#                  ,where=da_elnino_crit.notnull(),facecolor=ax1color, alpha=0.3, transform=trans)

#### setting the plotting format
ax1.set_ylabel('ONI ($^o$C)',{'size':'20'},color='k')
ax1.set_xlabel('Year',{'size':'20'})
ax1.tick_params(axis='y',labelsize=20,labelcolor='k')
ax1.tick_params(axis='x',labelsize=20,labelcolor='k')
ax1.set_title("%s"%('ONI index'),{'size':'24'},pad=24)
ax1.legend(loc='upper left',fontsize=14)
ax1.grid(linestyle='dashed')

fig.savefig('./figure/figure20b.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)
