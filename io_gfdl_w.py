#!/usr/bin/env python
# coding: utf-8

# # Calculate vertical velocity field
#
#     The script utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other

import os
import dask
import xarray as xr
import numpy as np
import time as te
from mem_track import used_memory

import warnings
warnings.simplefilter("ignore")

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=6, processes=False)

########################################################## CORE
#### possible input info from external text file
## Read OMODEL dataset
# constant setting
syear = 1948
fyear = 1967
tp_lat_region = [-50,50]     # extract model till latitude

Model_varname = ['uo','vo','thetao']
Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
Area_name = ['areacello_cu','areacello_vu','areacello']
regridder_name = ['%s2t'%var for var in Model_varname]

Model_name = ['CORE']

# standard model (interpolated to this model)
Model_standard = 'CORE'

# inputs
modelin = {}
model = Model_name[0]
modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
modelfile = [['CORE_uo_1948_1967.zarr'],
             ['CORE_vo_1948_1967.zarr'],
             ['CORE_thetao_1948_1967.zarr']]

########################################################## JRA
# #### possible input info from external text file
# ## Read OMODEL dataset
# # constant setting
# syear = 1958
# fyear = 2017
# tp_lat_region = [-50,50]     # extract model till latitude

# Model_varname = ['uo','vo','thetao']
# Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
# Area_name = ['areacello_cu','areacello_vu','areacello']
# regridder_name = ['%s2t'%var for var in Model_varname]

# Model_name = ['JRA']

# # standard model (interpolated to this model)
# Model_standard = 'JRA'

# # inputs
# modelin = {}
# model = Model_name[0]
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_uo.zarr'],
#              ['JRA_vo.zarr'],
#              ['JRA_thetao.zarr']]


############################################################################
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
#      (!!!!!!!! remove all previous read model info if exec !!!!!!!!!!)
nmodel = len(Model_name)
nvar = len(Model_varname)

ds_model_mlist = {}
mean_mlist = {}
season_mlist = {}


# Removing mean and seasonal signal
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
        if var not in [Tracer_varname]:
            regridder = xe.Regridder(ds_model_mlist[model][var],                    
                                     ds_model_mlist[Model_standard][Tracer_varname],                           
                                     'bilinear',filename='%s2t_%s.nc'%(var,model),
                                     periodic=True,reuse_weights=True)
            regridder_list['%s2t'%(var)] = regridder
    regridder_mlist[model] = regridder_list


# # Derived field
import numpy as np

def cal_delta_distance(da_var):
    """
    calculate grid cell distance (dx/dy) based on lon lat location
    """
    r_earth = 6.371E8         # cm

    da_dx = da_var.lon.copy()
    delx = np.abs(da_var.lon.values[:,1:]-da_var.lon.values[:,:-1])
    da_dx.values[:,1:-1] = (delx[:,:-1]+delx[:,1:])/2.
    da_dx.values[:,0] = delx[:,0]
    da_dx.values[:,-1] = delx[:,-1]
    da_dx = da_dx/180.*np.pi*r_earth*np.cos(da_var.lat/180.*np.pi)/100.   # m

    da_dy = da_var.lat.copy()
    dely = np.abs(da_var.lat.values[1:,:]-da_var.lat.values[:-1,:])
    da_dy.values[1:-1,:] = (dely[:-1,:]+dely[1:,:])/2.
    da_dy.values[0,:] = dely[0,:]
    da_dy.values[-1,:] = dely[-1,:]
    da_dy = da_dy/180.*np.pi*r_earth/100.    # m

    return da_dx, da_dy


def w_continuity(da_u,da_v):
    """
    Calculate the vertical velocity based on the continuity (incompressible)
    in the column integrated column from the surface to a certain level.

    input :

    da_u: xr.DataArray of x-direction velocity in 3D
    da_v: xr.DataArray of y-direction velocity in 3D

    """

    # calculate delta u
    da_du = da_u.copy()
    delu = da_u.diff('x').values[:,:,1:]+da_u.diff('x').values[:,:,:-1]
    da_du[:,:,1:-1] = delu[:,:,:]/2.
    da_du[:,:,0] = da_u.diff('x').values[:,:,0]
    da_du[:,:,-1] = da_u.diff('x').values[:,:,-1]

    # calculate delta v
    da_dv = da_v.copy()
    delv = da_v.diff('y').values[:,1:,:]+da_v.diff('y').values[:,:-1,:]
    da_dv[:,1:-1,:] = delv[:,:,:]/2.
    da_dv[:,0,:] = da_v.diff('y').values[:,0,:]
    da_dv[:,-1,:] = da_v.diff('y').values[:,-1,:]

    # calculate delta x/y
    da_dx, da_dy = cal_delta_distance(da_du)  # m

    # calculate unit area column integrated divergence
    da_div = da_du/da_dx+da_dv/da_dy           # 1/s

    da_w = da_u.copy()
    for zind, zz in enumerate(da_div.z) :
        da_div_colint = da_div[:zind+1,:,:].integrate(dim='z')  # m/s
        da_w[zind,:,:] = da_div_colint   # assum w=0 at surface layer
                                         # (sign might vary in different model due to z coord)

    return da_w, da_div


# initialize dictionary  (exec this cell will remove all previous calculated values)
w_mlist={}

for nmodel,model in enumerate(Model_name):
    w_mlist[model]={}


for nmodel,model in enumerate(Model_name):
    
    # Model
    model      = Model_name[nmodel]
    crit_dep   = 1000.

    #### output dir
    modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/%s/derived_field/'%Model_name[nmodel]
    modelfile = '%s_wo_scpt/'%Model_name[nmodel]
    
    for tt,time in enumerate(ds_model_mlist[model]['uo'].time):
        stime = te.time()
        print('============ Time',time.values,'===========')
        ds = xr.Dataset()

        da_uo = ds_model_mlist[model]['uo'].isel(time=tt)#+mean_mlist[model]['uo']
        da_vo = ds_model_mlist[model]['vo'].isel(time=tt)#+mean_mlist[model]['vo']
        
        # crop both array based on set critical deptj
        da_uo = da_uo.where(da_uo.z <= crit_dep,drop=True)
        da_vo = da_vo.where(da_vo.z <= crit_dep,drop=True)

        # regrid u,v to tracer point
        da_uo_all = regridder_mlist[model]['uo2t'](da_uo)
        da_vo_all = regridder_mlist[model]['vo2t'](da_vo)

        # calculate vertical velocity based on continuity eq (incompressible)
        da_wo_all,  da_div_all = w_continuity(da_uo_all,da_vo_all)

        ds['wo'] = da_wo_all
        ds['div'] = da_div_all

        if not os.path.exists(modeldir+modelfile):
            os.makedirs(modeldir+modelfile)
        ds.to_netcdf(modeldir+modelfile+str(time.values)[:7]+'.nc', mode='w')
        
        ftime = te.time()
        used_memory()
        print("calculation time:",(ftime-stime)/60.,'mins')
