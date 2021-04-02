#!/usr/bin/env python
# coding: utf-8
#
# conda env => update_python3 (for updated dask package)

# # Calculate thermocline field  
#
#     The notebook utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other
#
# Different approach of thermocline(some called mix layer depth which is less accurate) approach can be used
# 1. true temperature (potential temp) gradient of 0.02C/m [Wyrtki, 1964]
# 2. temperature difference from the surface is 0.5C [Wyrtki, 1964]
# 3. density difference from the surface is 0.125 kg/m^3 [Levitus, 1982] => more like mixed layer
# 3. density gradient of 0.01 kg/m^4 [Lukas et al., 1991] => more like mixed layer
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
client = Client(n_workers=1, threads_per_worker=6, processes=False)

print('dashboard @ port : %i'%(client.scheduler_info()['services']['bokeh']))


# # Read OMODEL dataset
#
# read in as dask array to avoid memory overload
import warnings
warnings.simplefilter("ignore")


# OMODEL file detail

######################################## CORE ###########################################
#### possible input info from external text file
# constant setting
syear = 1948
fyear = 1967
tp_lat_region = [-50,50]     # extract model till latitude

Model_varname = ['thetao','so']
Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
Area_name = ['areacello','areacello']
regridder_name = ['%s2t'%var for var in Model_varname]

Model_name = ['CORE']

# standard model (interpolated to this model)
Model_standard = 'CORE'

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
modelfile = [['CORE_thetao_1948_1967.zarr'],
             ['CORE_so_1948_1967.zarr']]
# Model
crit_dep   = 500

#### output dir
outdir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/derived_field/'
outfile = 'CORE_mixed_layer_rho_scpt/'

# ######################################### JRA ###########################################

# # # OMODEL file detail
# #### possible input info from external text file
# # constant setting
# syear = 1958
# fyear = 2017
# tp_lat_region = [-50,50]     # extract model till latitude

# Model_varname = ['thetao','so']
# Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
# Area_name = ['areacello','areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]

# Model_name = ['JRA']

# # standard model (interpolated to this model)
# Model_standard = 'JRA'

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_thetao.zarr'],['JRA_so.zarr']]

# # Model
# crit_dep   = 500

# #### output dir
# outdir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/derived_field/'
# outfile = 'JRA_mixed_layer_rho_scpt/'




########################### main program ##################################
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
#        (!!!!!!!! remove all previous read model info if exec !!!!!!!!!!)
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
                
                # for some old file include var: pacific
                try :
                    ds_model_sub = ds_model_sub.drop('pacific')
                except ValueError:
                    print('')
                        
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
        .where((ds_model.lat >= np.min(np.array(tp_lat_region)))&\
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




# # Create regridder for all var to tracer points (standard model)
import xesmf as xe

# Regridding to the tracer points
regridder_mlist = {}
for nmodel,model in enumerate(Model_name):
    regridder_list = {}
    for nvar,var in enumerate(Model_varname):
        regridder = xe.Regridder(ds_model_mlist[model][var],\
                                 ds_model_mlist[Model_standard][Tracer_varname],\
                                 'bilinear',filename='%s2t_%s.nc'%(var,model),periodic=True,reuse_weights=True)
        regridder_list['%s2t'%(var)] = regridder
    regridder_mlist[model] = regridder_list



# # Derived field
import gsw


def z_to_p(da_z,da_lat):
    """
    input: Depth, positive up in meter
    output: Pressure dbar
            (( i.e. absolute pressure - 10.1325 dbar ))
    """
    da_p=xr.apply_ufunc(gsw.conversions.p_from_z,
                        da_z,da_lat,
                        input_core_dims=[[],[]],
                        vectorize=True,
                        kwargs={'geo_strf_dyn_height':0,
                                'sea_surface_geopotential':0})
                        #dask='allowed')
    return da_p

def SP_to_SA(da_sp,da_p,da_lon,da_lat):
    """
    input: Practical Salinity (PSS-78)
           Longitude, -360 to 360 degrees
           Latitude, -90 to 90 degrees
    output: Absolute Salinity (g/kg)
    """
    da_sa=xr.apply_ufunc(gsw.conversions.SA_from_SP,
                        da_sp,da_p,da_lon,da_lat,
                        input_core_dims=[[],[],[],[]],
                        vectorize=True)
                        #dask='allowed')
    return da_sa

def pt_to_CT(da_sa,da_pt):
    """
    input: Potential temp (degrees C)
    output: Conservative Temperature (ITS-90)
    """
    da_ct=xr.apply_ufunc(gsw.conversions.CT_from_pt,
                        da_sa,da_pt,
                        input_core_dims=[[],[]],
                        vectorize=True)
                        #dask='allowed')
    return da_ct

def teos10_potential_rho(da_sa,da_ct):
    """
    input: Absolute Salinity (g/kg)
           Conservative Temperature (ITS-90, degrees C)
           Sea pressure (absolute pressure minus 10.1325 dbar)
    output: Potential density (kg/m^3) anomaly with respect to 0dbar (potential density - 1000 kg/m^3)
    """
    da_sigma0_ano=xr.apply_ufunc(gsw.density.sigma0,
                        da_sa,da_ct,
                        input_core_dims=[[],[]],
                        vectorize=True)
                        #dask='allowed')
    return da_sigma0_ano


def mix_layer_lukas1991(da_sigma0_ano):
    """
    density vertical gradient threshold to find mixed layer depth
    density gradient threshold of 0.01 kg/m^4

    input :

    da_sigma0_ano: xr.DataArray of potential density (kg/m^3) in 3D

    """

    # make land mask based on surface layer
    da_mask = da_sigma0_ano.isel(z=0)*0.+1.

    # calculate drho/dz
    da_dsigma0_dz = da_sigma0_ano.differentiate('z') # kg/m^4

    # interpolate to finer vertical resolution (5m)
    da_interp = da_dsigma0_dz.interp(z=np.arange(0,da_dsigma0_dz.z.max(),2.5))

    # remove values shallower than critcal value
    da_interp_masked = da_interp.where(da_interp>0.01,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_interp_masked.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_mld = da_interp.z[z_ind]*da_mask

    return da_mld

def mix_layer_levitus1982(da_sigma0_ano):
    """
    density difference from surface threshold to find mixed layer depth
    density difference threshold of 0.125 kg/m^3

    input :

    da_sigma0_ano: xr.DataArray of potential density (kg/m^3) in 3D

    """

    # interpolate to finer vertical resolution (5m)
    da_interp = da_sigma0_ano.interp(z=np.arange(0,da_sigma0_ano.z.max(),2.5))

    # make land mask based on surface layer
    da_mask = da_sigma0_ano.isel(z=0)*0.+1.

    # calculate rho-rho0
    da_diff = np.abs(da_interp-da_sigma0_ano.isel(z=0))

    # remove values shallower than critcal value
    da_diff = da_diff.where(da_diff>0.125,other=99999)

    # find first index that have value larger than critical value
    z_ind = da_diff.argmin(dim='z',skipna=True)

    # used 2d index to find 2d depth map
    da_mld = da_diff.z[z_ind]*da_mask

    return da_mld

# initialize dictionary  (exec this cell will remove all previous calculated values)
mixeddep_mlist={}

for nmodel,model in enumerate(Model_name):
    mixeddep_mlist[model]={}


for nmodel,model in enumerate(Model_name):
    for tt,time in enumerate(ds_model_mlist[model]['thetao'].time):
        ds = xr.Dataset()

        da_thetao = ds_model_mlist[model]['thetao'].isel(time=tt).load() #+mean_mlist[model]['thetao']).compute()
        da_so = ds_model_mlist[model]['so'].isel(time=tt).load() #+mean_mlist[model]['so']).compute()

#         da_thetao = da_thetao.persist()
#         da_so = da_so.persist()

        # crop both array based on set critical deptj
        da_thetao = da_thetao.where(da_thetao.z <= crit_dep,drop=True)
        da_so = da_so.where(da_so.z <= crit_dep,drop=True)

        # change depth to pressure
        da_p = z_to_p(-da_thetao.z,da_thetao.lat[:,0])

        # change psu to absolute salinity
        da_sa = SP_to_SA(da_so,da_p,da_so.lon,da_so.lat)

        # change potential temp to conservative temp
        da_ct = pt_to_CT(da_sa,da_thetao)

        # calculate density using absolute salinity and conservative temp
        da_sigma0 = teos10_potential_rho(da_sa,da_ct)

        # calculate gradient rho to determine mixed layer depth
        da_mld = mix_layer_levitus1982(da_sigma0)

        ds['potential_rho'] = da_sigma0
        ds['mld'] = da_mld

        del ds.z.attrs['edges']
        if not os.path.exists(outdir+outfile):
            os.makedirs(outdir+outfile)
        ds.to_netcdf(outdir+outfile+str(time.values)[:7]+'.nc', mode='w')
