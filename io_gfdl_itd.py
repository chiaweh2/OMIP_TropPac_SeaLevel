#!/usr/bin/env python
# coding: utf-8

# # Calculate thermocline field
#
#     The notebook utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other
#
# Different approach of thermocline(some called mix layer depth which is less accurate) approach can be used
# 1. true temperature (potential temp) gradient of 0.02C/m [Wyrtki, 1964]
# 2. temperature difference from the surface is 0.5C [Wyrtki, 1964]
# 3. temperature equal to 20 degree
#
#

# In[1]:


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
client = Client(n_workers=1, threads_per_worker=8, processes=False)
client


from mem_track import used_memory
used_memory()

# # Read OMODEL dataset
#
# read in as dask array to avoid memory overload

import warnings
warnings.simplefilter("ignore")


# # OMODEL file detail
#### possible input info from external text file
# constant setting
syear = 1958
fyear = 2017
tp_lat_region = [-50,50]     # extract model till latitude

Model_varname = ['thetao']
Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
Area_name = ['areacello']
regridder_name = ['%s2t'%var for var in Model_varname]

Model_name = ['JRA']

# standard model (interpolated to this model)
Model_standard = 'JRA'

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
modelfile = [['JRA_thetao.zarr']]

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


# initialization of dict and list  (!!!!!!!! remove all previous read model info if exec !!!!!!!!!!)
nmodel = len(Model_name)
nvar = len(Model_varname)

ds_model_mlist = {}
mean_mlist = {}
season_mlist = {}


# # Removing mean and seasonal signal

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

#         # calculate mean
#         mean_list[var] = ds_model_list[var].mean(dim='time').compute()
#         ds_model_list[var] = ds_model_list[var]-mean_list[var]

#         # calculate seasonality
#         season_list[var] = ds_model_list[var].groupby('time.month').mean(dim='time').compute()
#         ds_model_list[var] = ds_model_list[var].groupby('time.month')-season_list[var]

#     mean_mlist[model] = mean_list
#     season_mlist[model] = season_list
    ds_model_mlist[model] = ds_model_list


# # Derived field

def isothermal_depth_wyrtki1964_gradient(da_PT):
    """
    potential temperature vertical gradient threshold to find isothermal layer depth
    temperature gradient threshold of 0.02 kg/m^4

    input :

    da_PT: xr.DataArray of potential temperature (degree C) in 3D

    """

    # make land mask based on surface layer
    da_mask = da_PT.isel(z=0)*0.+1.

    # calculate drho/dz
    da_PT_dz = da_PT.differentiate('z') # kg/m^4

    # interpolate to finer vertical resolution (2.5m)
    da_interp = da_PT_dz.interp(z=np.arange(0,da_PT_dz.z.max(),2.5))

    # remove values shallower than critcal value
    da_interp_masked = da_interp.where(da_interp>0.02,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_interp_masked.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_itd = da_interp.z[z_ind]*da_mask

    return da_itd

def isothermal_depth_wyrtki1964(da_PT):
    """
    potential temperature difference from surface threshold to find isothermal layer depth
    temperature difference threshold of 0.5 degree C

    input :

    da_PT: xr.DataArray of potential temperature (degree C) in 3D

    """

    # interpolate to finer vertical resolution (2.5m)
    da_interp = da_PT.interp(z=np.arange(0,da_PT.z.max(),2.5))

    # make land mask based on surface layer
    da_mask = da_PT.isel(z=0)*0.+1.

    # calculate rho-rho0
    da_diff = np.abs(da_interp-da_PT.isel(z=0))

    # remove values shallower than critcal value
    da_diff = da_diff.where(da_diff>0.5,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_diff.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_itd = da_diff.z[z_ind]*da_mask

    return da_itd

def isothermal_depth_wyrtki1964_gradient(da_PT):
    """
    potential temperature vertical gradient threshold to find isothermal layer depth
    temperature gradient threshold of 0.02 kg/m^4

    input :

    da_PT: xr.DataArray of potential temperature (degree C) in 3D

    """

    # make land mask based on surface layer
    da_mask = da_PT.isel(z=0)*0.+1.

    # calculate drho/dz
    da_PT_dz = da_PT.differentiate('z') # kg/m^4

    # interpolate to finer vertical resolution (2.5m)
    da_interp = da_PT_dz.interp(z=np.arange(0,da_PT_dz.z.max(),2.5))

    # remove values shallower than critcal value
    da_interp_masked = da_interp.where(da_interp>0.02,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_interp_masked.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_itd = da_interp.z[z_ind]*da_mask

    return da_itd


def isothermal_depth_d20(da_PT):
    """
    potential temperature equal to 20 degree of isothermal layer depth

    input :

    da_PT: xr.DataArray of potential temperature (degree C) in 3D

    """

    # interpolate to finer vertical resolution (2.5m)
    da_interp = da_PT.interp(z=np.arange(0,da_PT.z.max(),2.5))

    # make land mask based on surface layer
    da_mask = da_PT.isel(z=0)*0.+1.

    # remove values smaller than critcal value
    da_interp = da_interp.where(da_interp>20,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_interp.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_itd = da_interp.z[z_ind]*da_mask

    return da_itd

# initialize dictionary  (exec this cell will remove all previous calculated values)
mixeddep_mlist={}

for nmodel,model in enumerate(Model_name):
    mixeddep_mlist[model]={}


# Model
model      = 'JRA'
crit_dep   = 500

#### output dir
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/derived_field/'
modelfile = 'JRA_d20_layer/'


for nmodel,model in enumerate(Model_name):
    for tt,time in enumerate(ds_model_mlist[model]['thetao'].time):
        ds = xr.Dataset()

        da_thetao = ds_model_mlist[model]['thetao'].isel(time=tt)#+mean_mlist[model]['thetao']
        da_thetao = da_thetao.compute()

        # crop both array based on set critical deptj
        da_thetao = da_thetao.where(da_thetao.z <= crit_dep,drop=True)

        # calculate gradient rho to determine mixed layer depth
        da_itd = isothermal_depth_d20(da_thetao)

        ds['itd'] = da_itd

        if not os.path.exists(modeldir+modelfile):
            os.makedirs(modeldir+modelfile)
        ds.to_netcdf(modeldir+modelfile+str(time.values)[:7]+'.nc', mode='w')
