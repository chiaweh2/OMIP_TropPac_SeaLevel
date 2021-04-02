#!/usr/bin/env python
# coding: utf-8

# Output transect data for faster processing
#     The notebook utilized the new zarr format for faster IO and dask array processing.
#     The output is in netcdf format for easy sharing with other
#
# (1) cropped dataset based on time and transect
# (2) regrid the different variable to the same grid


import os
import dask
import xarray as xr
import numpy as np
import sys
from dask.distributed import Client
import xesmf as xe
import time as te

import warnings
warnings.simplefilter("ignore")

client = Client(n_workers=1, threads_per_worker=8, processes=False)

from mem_track import used_memory
used_memory()


#### possible input info from external text file
## OMODEL file detail


# #################### 
# # standard model (interpolated to this model)
# Model_standard = 'CORE'

# syear = 1968
# fyear = 2007
# Model_name = ['CORE']
# modelin = {}
# model = Model_name[0]
# modeldir = './data/GFDL/CORE/'
# modelfile = [['CORE_uo_1948_1967.zarr',CORE_uo_1968_1992.zarr','CORE_uo.zarr'],
#              ['CORE_thetao_1948_1967.zarr','CORE_thetao_1968_1992.zarr','CORE_thetao.zarr'],
#              ['CORE_vo_1948_1967.zarr','CORE_vo_1968_1992.zarr','CORE_vo.zarr'],
#              ['derived_field/CORE_wo_scpt/']]
# Model_varname = ['uo','thetao','vo','wo']
# Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
# derived_field = ['wo']
# no_regrid = ['wo','thetao']

# # output transect
# transect_plane = ['yz','yz','yz','xz','xz','xz','xy']
# transect_var = [['uo','thetao'],
#                 ['uo','thetao'],
#                 ['uo','thetao'],
#                 ['vo','thetao'],
#                 ['vo','thetao'],
#                 ['uo','thetao','vo'],
#                 ['wo','thetao']]
# transect_x   = [[-160],[-180],[120],[],[],[],[]]
# transect_y   = [[],[],[],[-20],[20],[0],[]]
# transect_z   = [[],[],[],[],[],[],[400]]


# # outputs
# modelout = {}
# modeldir_out = './data/GFDL/CORE/transect_scpt/'
# modelfile_out = ['CORE_uo_thetao',
#                  'CORE_uo_thetao',
#                  'CORE_uo_thetao',
#                  'CORE_vo_thetao',
#                  'CORE_vo_thetao',
#                  'CORE_uo_thetao_vo',
#                  'CORE_wo_thetao']


#################### 
# standard model (interpolated to this model)
Model_standard = 'JRA'

syear = 1958
fyear = 2017
Model_name = ['JRA']
modelin = {}
model = Model_name[0]
modeldir = './data/GFDL/JRA/'
modelfile = [['JRA_uo.zarr'],['JRA_thetao.zarr'],['JRA_vo.zarr'],['derived_field/JRA_wo_scpt/']]
Model_varname = ['uo','thetao','vo','wo']
Tracer_varname = 'thetao'         # the variable name at the tracer point (Arakawa C grid)
derived_field = ['wo']
no_regrid = ['wo','thetao']

# output transect
transect_plane = ['yz','yz','yz','xz','xz','xz','xy']
transect_var = [['uo','thetao'],
                ['uo','thetao'],
                ['uo','thetao'],
                ['vo','thetao'],
                ['vo','thetao'],
                ['uo','thetao','vo'],
                ['wo','thetao']]
transect_x   = [[-160],[-180],[120],[],[],[],[]]
transect_y   = [[],[],[],[-20],[20],[0],[]]
transect_z   = [[],[],[],[],[],[],[400]]


# outputs
modelout = {}
modeldir_out = './data/GFDL/JRA/transect_scpt/'
modelfile_out = ['JRA_uo_thetao',
                 'JRA_uo_thetao',
                 'JRA_uo_thetao',
                 'JRA_vo_thetao',
                 'JRA_vo_thetao',
                 'JRA_uo_thetao_vo',
                 'JRA_wo_thetao']




#####################################################################
# # Read OMODEL dataset
# input
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

# output
for nmodel,model in enumerate(Model_name):
    multifile = []
    for ntran,plane in enumerate(transect_plane):
        if plane in ['yz']:
            multifile.append(os.path.join(modeldir_out,modelfile_out[ntran]\
                                          +'_%s_%i'%(transect_plane[ntran],transect_x[ntran][0])))
        elif plane in ['xz']:
            multifile.append(os.path.join(modeldir_out,modelfile_out[ntran]\
                                          +'_%s_%i'%(transect_plane[ntran],transect_y[ntran][0])))
        elif plane in ['xy']:
            multifile.append(os.path.join(modeldir_out,modelfile_out[ntran]\
                                          +'_%s_%i'%(transect_plane[ntran],transect_z[ntran][0])))

    modelout[model] = multifile


# initialization of dict and list
nmodel = len(Model_name)
nvar = len(Model_varname)
ds_model_mlist = {}
mean_mlist = {}
season_mlist = {}


# # Removing mean and seasonal signal
for nmodel,model in enumerate(Model_name):
    ds_model_list = {}
    mean_list = {}
    season_list = {}
    for nvar,var in enumerate(Model_varname):
        print('read %s %s'%(model,var))

        # read input data
        if var not in derived_field:
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
        else:
            ds_model = xr.open_mfdataset(os.path.join(modelin[model][nvar][0],'????-??.nc'),
                                         concat_dim='time',
                                         chunks={'z':-1,'y':100,'x':100})
            
            # set time dimension
            filenames_all = os.listdir(modelin[model][nvar][0])   
            filenames_all.sort()
            filenames_date = [np.datetime64(file[:7]) for file in filenames_all if "._" not in file] 
            ds_model = ds_model.assign_coords(time = filenames_date)

        # crop data (time)
        da_model = ds_model[var].where((ds_model['time.year'] >= syear)&\
                                       (ds_model['time.year'] <= fyear),drop=True)

        #da_model = da_model.where((ds_model.lat >= np.min(np.array(tp_lat_region)))&\
        #                          (ds_model.lat <= np.max(np.array(tp_lat_region))),drop=True)

        # store all model data
        ds_model_list[var] = da_model

        # # calculate mean
        # mean_list[var] = ds_model_list[var].mean(dim='time').compute()
        # ds_model_list[var] = ds_model_list[var]-mean_list[var]

        # # calculate seasonality
        # season_list[var] = ds_model_list[var].groupby('time.month').mean(dim='time').compute()
        # ds_model_list[var] = ds_model_list[var].groupby('time.month')-season_list[var]

    # mean_mlist[model] = mean_list
    # season_mlist[model] = season_list
    ds_model_mlist[model] = ds_model_list




for nmodel,model in enumerate(Model_name):
    for tt,time in enumerate(ds_model_mlist[model][Tracer_varname].time):
        stime = te.time()
        print('============ Time',time.values,'===========')
        
        list_ds_transect = []
        for ntran,plane in enumerate(transect_plane):
            list_ds_transect.append(xr.Dataset())

        # different variable   
        for var in Model_varname:
            print('var:',var)
            if var not in no_regrid:
                regridder1 = xe.Regridder(ds_model_mlist[model][var],
                            ds_model_mlist[Model_standard][Tracer_varname],
                            'bilinear',
                            filename='%s2t_%s.nc'%(var,model),
                            periodic=True,
                            reuse_weights=True)
                da_regrid = regridder1(ds_model_mlist[model][var].isel(time=tt))
                used_memory()
                print('regridded')
                da_regrid = da_regrid.assign_coords(x = ds_model_mlist[model][Tracer_varname].x)
                da_regrid = da_regrid.assign_coords(y = ds_model_mlist[model][Tracer_varname].y)
            else:
                da_regrid = ds_model_mlist[model][var].isel(time=tt)
                    
            # different transect
            for ntran,plane in enumerate(transect_plane):
                # correct the lon range
                lon_mod = np.array(transect_x[ntran])
                lonmin = ds_model_mlist[model][Tracer_varname].lon.min()
                ind1 = np.where(lon_mod>np.float(360.+lonmin))[0]
                # change Lon range to -300-60 (might be different for different model)
                lon_mod[ind1] = lon_mod[ind1]-360.
                transect_x[ntran] = lon_mod
                if var in transect_var[ntran]:
                    if plane in ['yz']:
                        print("processing",plane,transect_x[ntran])
                        list_ds_transect[ntran][var] = da_regrid.sel(x=transect_x[ntran], method='nearest')
                    elif plane in ['xz']:
                        print("processing",plane,transect_y[ntran])
                        list_ds_transect[ntran][var] = da_regrid.sel(y=transect_y[ntran], method='nearest')
                    elif plane in ['xy']:
                        print("processing",plane,transect_z[ntran])
                        list_ds_transect[ntran][var] = da_regrid.sel(z=transect_z[ntran], method='nearest')
                
        for ntran,plane in enumerate(transect_plane):         
#             del list_ds_transect[ntran].z.attrs['edges']
            if not os.path.exists(modelout[model][ntran]):
                os.makedirs(modelout[model][ntran])
            try :
                os.remove(modelout[model][ntran]+'/'+str(time.values)[:7]+'.nc')
                list_ds_transect[ntran].to_netcdf(modelout[model][ntran]+'/'+str(time.values)[:7]+'.nc', mode='w')
            except FileNotFoundError:
                list_ds_transect[ntran].to_netcdf(modelout[model][ntran]+'/'+str(time.values)[:7]+'.nc', mode='w')
            
        ftime = te.time()
        used_memory()
        print("calculation time:",(ftime-stime)/60.,'mins')
            
            

                    
            