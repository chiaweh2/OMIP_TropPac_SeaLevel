#!/usr/bin/env python
# coding: utf-8

# # Regional average
#
#     The script utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other
#

import os
import cftime
import dask
import xarray as xr
import numpy as np
import nc_time_axis
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from create_ocean_mask import levitus98

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=8, processes=False)

from mem_track import used_memory
used_memory()


# # # OMODEL file detail
#############################################################################
# #### possible input info from external text file
# # constant setting
# syear = 1948
# fyear = 2007
# tp_lat_region = [-50,50]     # extract model till latitude

# Model_varname = ['hflso','hfsso','rlntds','rsntds']
# Area_name = ['areacello','areacello','areacello','areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]

# Model_name = ['CORE']

# # standard model (interpolated to this model)
# Model_standard = 'CORE'

# # regional mean range
# lon_range_list = [[120,180],[180,-60]]    # Lon: -180-180
# lat_range_list = [[-20,20],[-20,20]]

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
# modelfile = [['CORE_hflso.zarr'],['CORE_hfsso.zarr'],['CORE_rlntds.zarr'],['CORE_rsntds.zarr']]


# # output dir
# dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/regional_avg/'


#############################################################################
#### possible input info from external text file
# constant setting
syear = 1958
fyear = 2017
tp_lat_region = [-50,50]     # extract model till latitude

Model_varname = ['hflso','hfsso','rlntds','rsntds']
Area_name = ['areacello','areacello','areacello','areacello']
regridder_name = ['%s2t'%var for var in Model_varname]

Model_name = ['JRA']

# standard model (interpolated to this model)
Model_standard = 'JRA'

# regional mean range
lon_range_list = [[120,180],[180,-60]]    # Lon: -180-180
lat_range_list = [[-20,20],[-20,20]]

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
modelfile = [['JRA_hflso.zarr'],['JRA_hfsso.zarr'],['JRA_rlntds.zarr'],['JRA_rsntds.zarr']]


# output dir
dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/regional_avg/'



# #############################################################################
# #### possible input info from external text file
# # constant setting
# syear = 1948
# fyear = 2007
# tp_lat_region = [-50,50]     # extract model till latitude

# Model_varname = ['net_heat_coupler']
# Area_name = ['areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]

# Model_name = ['CORE']

# # standard model (interpolated to this model)
# Model_standard = 'CORE'

# # regional mean range
# lon_range_list = [[120,180],[180,-60]]    # Lon: -180-180
# lat_range_list = [[-20,20],[-20,20]]

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
# modelfile = [['CORE_net_heat_coupler.zarr']]


# # output dir
# dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/regional_avg/'


##############################################################################
# #### possible input info from external text file
# # constant setting
# syear = 1958
# fyear = 2017
# tp_lat_region = [-50,50]     # extract model till latitude

# Model_varname = ['net_heat_coupler']
# Area_name = ['areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]

# Model_name = ['JRA']

# # standard model (interpolated to this model)
# Model_standard = 'JRA'

# # regional mean range
# lon_range_list = [[120,180],[180,-60]]    # Lon: -180-180
# lat_range_list = [[-20,20],[-20,20]]

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_net_heat_coupler.zarr']]


# # output dir
# dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/regional_avg/'




################################################################
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
                         .where((ds_model['time.year'] >= syear)&
                                (ds_model['time.year'] <= fyear)
                                 ,drop=True)
        da_model = da_model\
                          .where((ds_model.lat >= np.min(np.array(tp_lat_region)))&
                                 (ds_model.lat <= np.max(np.array(tp_lat_region)))
                                 ,drop=True)

        # store all model data
        ds_model_list[var] = da_model

        # # calculate mean
        # mean_list[var] = ds_model_list[var].mean(dim='time').compute()
        # ds_model_list[var] = ds_model_list[var]-mean_list[var]
        #
        # # calculate seasonality
        # season_list[var] = ds_model_list[var].groupby('time.month').mean(dim='time').compute()
        # ds_model_list[var] = ds_model_list[var].groupby('time.month')-season_list[var]

    # mean_mlist[model] = mean_list
    # season_mlist[model] = season_list
    ds_model_mlist[model] = ds_model_list

used_memory()


# read mask dataset
ds_pac = levitus98(ds_model_mlist[model][Model_varname[0]],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='x',lat_name='y', new_regridder_name='')


# initialize dictionary  (exec this cell will remove all previous calculated values for all variables)
regional_var_mlist = {}
for nmodel,model in enumerate(Model_name):
    regional_var_mlist[model] = xr.Dataset()

for nmodel,model in enumerate(Model_name):
    for varname in Model_varname:
        varind = Model_varname.index(varname)
        for nn in range(len(lon_range_list)):
            print('process',lon_range_list[nn],lat_range_list[nn])

            #### setting individual event year range
            lon_range  = lon_range_list[nn]
            lat_range  = lat_range_list[nn]

            # correct the lon range
            lon_range_mod = np.array(lon_range)
            lonmin = ds_model_mlist[model][varname].lon.min()
            ind1 = np.where(lon_range_mod>np.float(360.+lonmin))[0]
            lon_range_mod[ind1] = lon_range_mod[ind1]-360. # change Lon range to -300-60 (might be different for different model)

            # crop region
            ds_mask = ds_pac.where(\
                                  (ds_pac.lon>=np.min(lon_range_mod))&\
                                  (ds_pac.lon<=np.max(lon_range_mod))&\
                                  (ds_pac.lat>=np.min(lat_range))&\
                                  (ds_pac.lat<=np.max(lat_range))\
                                   ).compute()

            # read areacello
            da_area = (xr.open_zarr(modelin[model][varind][0])[Area_name[varind]]*ds_mask).compute()

            # calculate regional mean
            regional_var_mlist[model]['%s_%i_%i_%i_%i'%(varname,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]\
            = ((ds_model_mlist[model][varname]*ds_mask*da_area).sum(dim=['x','y'])).compute()

            fileout = '%s_%s_regional_sum_ts_scpt.nc'%(model,varname)
            
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)
            try:
                regional_var_mlist[model]['%s_%i_%i_%i_%i'%(varname,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]\
                .to_netcdf(dir_out+fileout,mode='a')
            except FileNotFoundError:
                regional_var_mlist[model]['%s_%i_%i_%i_%i'%(varname,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]\
                .to_netcdf(dir_out+fileout,mode='w')
