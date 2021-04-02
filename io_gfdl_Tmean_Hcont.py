#!/usr/bin/env python
# coding: utf-8

# # Calculate heat content and mean temperature within the box
#   (modified from io_gfdl_Ttend.ipynb)
#
#     The scripts utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other
#

import os
import dask
import xarray as xr
import numpy as np
import sys

import warnings
warnings.simplefilter("ignore")

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=8, processes=False)

from mem_track import used_memory
used_memory()

from create_ocean_mask import levitus98

# OMODEL file detail


#################### CORE ######################
#### possible input info from external text file
syear = 1948
fyear = 2007
# tp_lat_region = [-20,20]     # extract model till latitude (seasonal and mean)

Model_varname = ['thetao']
Area_name = ['areacello']
Model_name = ['CORE']

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
modelfile = [['CORE_thetao_1948_1967.zarr','CORE_thetao_1968_1992.zarr','CORE_thetao.zarr']]

crit_dep   = 400
lon_range_list = [[120,180],[180,-60],[120,-160]]    # Lon: -180-180
lat_range_list = [[-20,20],[-20,20],[-20,20]]

#### output dir
dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/regional_avg/'
file_out1 = 'CORE_heatcont_scpt_%s.nc'%(crit_dep)
file_out2 = 'CORE_meantemp_scpt_%s.nc'%(crit_dep)

#################### JRA ######################
# #### possible input info from external text file
# syear = 1958
# fyear = 2017
# # tp_lat_region = [-20,20]     # extract model till latitude (seasonal and mean)

# Model_varname = ['thetao']
# Area_name = ['areacello']
# Model_name = ['JRA']

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_thetao.zarr']]

# crit_dep   = 400
# lon_range_list = [[120,180],[180,-60],[120,-160]]    # Lon: -180-180
# lat_range_list = [[-20,20],[-20,20],[-20,20]]

# #### output dir
# dir_out = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/regional_avg/'
# file_out1 = 'JRA_heatcont_scpt_%s.nc'%(crit_dep)
# file_out2 = 'JRA_meantemp_scpt_%s.nc'%(crit_dep)



#### constant
sea_heatcap = 3991.86795911963 # J/(kg K)
sea_density = 1025.       # kg/m^3
r_earth = 6.371E8         # cm

###############################################################
# store input file path
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


# initialization of dict and list
nmodel = len(Model_name)
nvar = len(Model_varname)

ds_model_mlist = {}
mean_mlist = {}
season_mlist = {}


## Read OMODEL dataset
# read in as dask array to avoid memory overload
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
                
                # for some old file include var: pacific
                try :
                    ds_model_sub = ds_model_sub.drop('pacific')
                except ValueError:
                    print('')
                        
                if nf == 0 :
                    ds_model = ds_model_sub
                else:
                    ds_model = xr.concat([ds_model,ds_model_sub],
                                         dim='time',
                                         data_vars='minimal')

        # crop data (time)
        da_model = ds_model[var].where((ds_model['time.year'] >= syear)&\
                                       (ds_model['time.year'] <= fyear),\
                                       drop=True)

        # da_model = da_model.where((ds_model.lat >= np.min(np.array(tp_lat_region)))&\
        #                           (ds_model.lat <= np.max(np.array(tp_lat_region))),\
        #                           drop=True)

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


# read mask dataset
ds_pac = levitus98(ds_model_mlist[model][Model_varname[0]],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='x',lat_name='y', new_regridder_name='')


## Derived heat content (over the entire region)

# initialize dictionary  (exec this cell will remove all previous calculated values)
heatcont_mlist={}
for nmodel,model in enumerate(Model_name):
    heatcont_mlist[model]={}


for nmodel,model in enumerate(Model_name):
    ds = xr.Dataset()
    ds_theta = xr.Dataset()
    for nn in range(len(lon_range_list)):
        print('process',lon_range_list[nn])

        #### setting individual event year range
        lon_range  = lon_range_list[nn]
        lat_range  = lat_range_list[nn]

        # correct the lon range
        mask_area_ind = Model_varname.index(Model_varname[0])
        lon_range_mod = np.array(lon_range)
        lonmin = ds_model_mlist[model][Model_varname[0]].lon.min()
        ind1 = np.where(lon_range_mod>np.float(360.+lonmin))[0]
        lon_range_mod[ind1] = lon_range_mod[ind1]-360.
        # change Lon range to -300-60 (might be different for different model)

        # crop region
        ds_mask = ds_pac.where(\
                              (ds_pac.lon>=np.min(lon_range_mod))&\
                              (ds_pac.lon<=np.max(lon_range_mod))&\
                              (ds_pac.lat>=np.min(lat_range))&\
                              (ds_pac.lat<=np.max(lat_range))\
                               ).compute()

        # read areacello
        da_area = (xr.open_zarr(modelin[model][mask_area_ind][0])[Area_name[mask_area_ind]]*ds_mask).compute()

        # crop depth
        ds_model_mlist[model]['thetao'] = ds_model_mlist[model]['thetao']\
                                          .where(ds_model_mlist[model]['thetao'].z <= crit_dep,drop=True)

        # calculate dz (meters)
        da_dz = ds_model_mlist[model]['thetao'].z.copy()
        da_dz.values[1:-1] = np.abs((ds_model_mlist[model]['thetao'].z[:-1].diff('z',1).values\
                                     +ds_model_mlist[model]['thetao'].z[1:].diff('z',1).values)/2.)
        da_dz.values[0] = np.abs((ds_model_mlist[model]['thetao'].z[1]\
                                  -ds_model_mlist[model]['thetao'].z[0]).values)
        da_dz.values[-1] = np.abs((ds_model_mlist[model]['thetao'].z[-1]\
                                   -ds_model_mlist[model]['thetao'].z[-2]).values)

        # calculate total volume
        vol_sum = (ds_mask*da_area*da_dz).sum()
        da_tot_thetao = ds_model_mlist[model]['thetao']*ds_mask*da_area*da_dz

        tv_sum_ts = da_tot_thetao.sum(dim=['x','y','z']).compute()
        da_heatcont_ts = tv_sum_ts*sea_heatcap*sea_density
        da_mean_thetao_ts = tv_sum_ts/vol_sum

        ds['heat_content_%i_%i'%(lon_range[0],lon_range[1])] = da_heatcont_ts
        ds_theta['mean_temp_%i_%i'%(lon_range[0],lon_range[1])] = da_mean_thetao_ts
        ds_theta['total_vol_%i_%i'%(lon_range[0],lon_range[1])] = vol_sum

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    try :
        os.remove(dir_out+file_out1)
        ds.to_netcdf(dir_out+file_out1, mode='w')
    except FileNotFoundError:
        ds.to_netcdf(dir_out+file_out1, mode='w') 
    try :
        os.remove(dir_out+file_out2)
        ds_theta.to_netcdf(dir_out+file_out2, mode='w')  
    except FileNotFoundError:
        ds_theta.to_netcdf(dir_out+file_out2, mode='w')