#!/usr/bin/env python
# coding: utf-8

# # Calculate the Oceanic Nino Index
# Nino3.4 index use the OMIP model output "tos" which is representing the
#  sea surface temperaturesea (SST) usually measured over the ocean.
#  Warm (red) and cold (blue) periods based on a threshold of +/- 0.5C for
#  the Oceanic Nino Index (ONI) [3 month running mean of ERSST.v5 SST anomalies
#  average over the Pacific Ocean tropic region in the Nino 3.4 region
#  (5N-5S, 120-170W)], based on centered 30-year base periods updated every 5 years.
#  (http://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)

import os
import cftime
import dask
import xarray as xr
import numpy as np
import nc_time_axis
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


import warnings
warnings.simplefilter("ignore")

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=1, processes=False)

from mem_track import used_memory
used_memory()

# # OMODEL file detail

######################################################################## JRA
#### possible input info from external text file
# constant setting
syear = 1948
fyear = 2007
tp_lat_region = [-50,50]     # extract model till latitude

Model_varname = ['tos']
Area_name = ['areacello']
regridder_name = ['%s2t'%var for var in Model_varname]

Model_name = ['CORE']

# standard model (interpolated to this model)
Model_standard = 'CORE'

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
modelfile = [['CORE_tos.zarr']]

#### output dir
outdir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/regional_avg/'


# ######################################################################## JRA
# #### possible input info from external text file
# # constant setting
# syear = 1958
# fyear = 2017
# tp_lat_region = [-50,50]     # extract model till latitude
#
# Model_varname = ['tos']
# Area_name = ['areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]
#
# Model_name = ['JRA']
#
# # standard model (interpolated to this model)
# Model_standard = 'JRA'
#
# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_tos.zarr']]
#
# #### output dir
# outdir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/regional_avg/'
# outfile = 'JRA_%s_oni_ts.nc'%(varname)

########################################################################
for nmodel,model in enumerate(Model_name):
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
# read in as dask array to avoid memory overload


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
                           .where((ds_model['time.year'] >= syear)&
                                 (ds_model['time.year'] <= fyear)
                                  ,drop=True)
        da_model = da_model\
                          .where((ds_model.lat >= np.min(np.array(tp_lat_region)))&
                                 (ds_model.lat <= np.max(np.array(tp_lat_region)))
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
varname = Model_varname[0]
lon_range_list = [[-170,-120]]    # Lon: -180-180
lat_range_list = [[-5,5]]        # Lat: -90-90

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
        da_area = da_area.where(
                              (da_area.lon>=np.min(lon_range_mod))&
                              (da_area.lon<=np.max(lon_range_mod))&
                              (da_area.lat>=np.min(lat_range))&
                              (da_area.lat<=np.max(lat_range))
                               ).compute()

        # calculate the temporal mean of regional mean
        mean_var = mean_mlist[model][varname]
        regional_mean = (mean_var*da_area).sum(dim=['x','y'])/(da_area).sum()

        # calculate time varying regional mean
        regional_var_mlist[model]['oni']\
             = ((ds_model_mlist[model][varname]*da_area).sum(dim=['x','y'])\
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

        outfile = '%s_%s_oni_ts.nc'%(model, varname)
        try:
            regional_var_mlist[model]['oni'].to_netcdf(outdir+outfile,mode='a')
        except FileNotFoundError:
            regional_var_mlist[model]['oni'].to_netcdf(outdir+outfile,mode='w')
