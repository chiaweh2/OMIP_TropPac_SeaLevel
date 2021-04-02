#!/usr/bin/env python

# # Calculate mean field


import os
import sys
import dask
import xarray as xr
import numpy as np

import warnings
warnings.simplefilter("ignore")

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=8, processes=False)
client


# # Read zarr
#############################################
# Model_name = ['JRA']
#
# # setting regional boundary
# tp_lat_region = np.array([-35,35])
#
# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_uo.zarr'],
#              ['JRA_vo.zarr'],
#              ['JRA_zos.zarr'],
#              ['JRA_tauuo.zarr'],
#              ['JRA_tauvo.zarr'],
#              ['JRA_tos.zarr'],
#              ['JRA_thetao.zarr'],
#              ['JRA_so.zarr'],
#              ['JRA_net_heat_coupler.zarr']]
# Model_varname = ['uo','vo','zos','tauuo','tauvo','tos','thetao','so','net_heat_coupler']
# Tracer_varname = 'zos'
#
# output_dir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/mean_field/'

# #############################################
# Model_name = ['JRA']
#
# # setting regional boundary
# tp_lat_region = np.array([-35,35])

# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/'
# modelfile = [['JRA_hflso.zarr'],
#              ['JRA_hfsso.zarr'],
#              ['JRA_rlntds.zarr'],
#              ['JRA_rsntds.zarr']]
# Model_varname = ['hflso','hfsso','rlntds','rsntds']
# Tracer_varname = 'hflso'
#
# output_dir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/JRA/mean_field/'

# ############################################
# Model_name = ['CORE']
# syear = 1948
# fyear = 2007
# # setting regional boundary
# tp_lat_region = np.array([-35,35])

# modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
# derivedir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/derived_field/'
# modelfile = [['CORE_uo_1968_1992.zarr','CORE_uo.zarr'],
#              ['CORE_vo_1968_1992.zarr','CORE_vo.zarr'],
#              ['CORE_zos.zarr'],
#              ['CORE_tauuo.zarr'],
#              ['CORE_tauvo.zarr'],
#              ['CORE_thetao_1968_1992.zarr','CORE_thetao.zarr'],
#              ['CORE_so_1968_1992.zarr','CORE_so.zarr'],
#              ['CORE_net_heat_coupler.zarr'],
#              ['CORE_wo_scpt']]
# Model_varname = ['uo','vo','zos','tauuo','tauvo','thetao','so','net_heat_coupler','wo']
# Tracer_varname = 'zos'

# output_dir1 = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/mean_field/'
# output_dir2 = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/seasonal_field/'


############################################
Model_name = ['CORE']
syear = 1948
fyear = 2007
# setting regional boundary
tp_lat_region = np.array([-35,35])

modeldir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/'
derivedir = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/derived_field/'
modelfile = [['CORE_wo_scpt']]
Model_varname = ['wo']
Tracer_varname = 'zos'

output_dir1 = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/mean_field/'
output_dir2 = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/GFDL/CORE/seasonal_field/'


####################### main program ###################
modelin = {}
for nmodel,model in enumerate(Model_name):
    multivar = []
    for file in modelfile :
        if file[0].endswith('.zarr'):
            if len(file) == 1 :
                multivar.append([os.path.join(modeldir,file[0])])
            elif len(file) > 1 :
                multifile = []
                for ff in file :
                    multifile.append(os.path.join(modeldir,ff))
                multivar.append(multifile)
        else:
            if len(file) == 1 :
                multivar.append([os.path.join(derivedir,file[0])])
            elif len(file) > 1 :
                print('Derived field should not have more than one dir location')
                sys.exit('Forced stop')
    modelin[model] = multivar


# initialization of dict and list
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
        if modelin[model][nvar][0].endswith('.zarr'):
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
            #-- single file
            if len(modelin[model][nvar]) == 1 :
                ds_model = xr.open_mfdataset(os.path.join(modelin[model][nvar][0],'????-??.nc'),
                                                      chunks={'y':1000, 'x':1000},
                                                      concat_dim='time')
                # set time dimension
                filenames_all = os.listdir(modelin[model][nvar][0])
                filenames_all.sort()
                filenames_date = [np.datetime64(file[:7]) for file in filenames_all if "._" not in file]
                ds_model = ds_model.assign_coords(time = filenames_date)
            elif len(modelin[model][nvar]) > 1 :
                print('Derived field should not have more than one dir location')


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

        # calculate mean
        mean_list[var] = ds_model_list[var].mean(dim='time').compute()

        # calculate seasonality
        season_list[var] = ds_model_list[var].groupby('time.month').mean(dim='time').compute()

        if not os.path.exists(output_dir1):
                os.makedirs(output_dir1)
        mean_list[var].to_netcdf(output_dir1+'%s_tro_1958_2017.nc'%(var))

        if not os.path.exists(output_dir2):
                os.makedirs(output_dir2)
        season_list[var].to_netcdf(output_dir2+'%s_tro_1958_2017.nc'%(var))
