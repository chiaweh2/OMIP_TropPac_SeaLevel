#!/usr/bin/env python
# coding: utf-8

# # figure surface flux
import os
import cftime
import dask
import xarray as xr
import numpy as np
import nc_time_axis
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


"""
The script generate the tropical surface net heat flux comparison
between model simulations
in the form of
figure 7 : spatial map
in the paper.

input files
============
JRA55-do : net_heat_coupler
CORE     : net_heat_coupler


function used
==================
create_ocean_mask.levitus98 : which generate the Pacific basin mask
spherical_area.cal_area     : generate area array based on the lon lat of data


"""


from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=8, processes=False)
client

import warnings
warnings.simplefilter("ignore")

from mem_track import used_memory
used_memory()


#### possible input info from external text file
# constant setting
syear = 1958
fyear = 2007
tp_lat_region = [-30,30]     # extract model till latitude

Model_varname = ['net_heat_coupler']
Area_name = ['areacello']

Model_name = ['JRA','CORE']
Model_legend_name = ['JRA55-do','CORE']

# standard model (interpolated to this model)
Model_standard = 'JRA'
Variable_standard = 'net_heat_coupler'
modeldir = './data/GFDL/JRA/'
modelfile = 'JRA_net_heat_coupler.zarr'
path_standard = modeldir+modelfile

# inputs
modelin = {}
path = {}
model = Model_name[0]
modeldir = './data/GFDL/JRA/'
modelfile = [['JRA_net_heat_coupler.zarr']]
path[model]=[modeldir,modelfile]

model = Model_name[1]
modeldir = './data/GFDL/CORE/'
modelfile = [['CORE_net_heat_coupler.zarr']]
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
                                 (ds_model['time.year'] <= fyear)\
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



# # Regridding
#### models
da_model_standard = xr.open_zarr(path_standard).isel(time=0).load()

da_model_standard = da_model_standard\
                  .where((da_model_standard.lat >= np.min(np.array(tp_lat_region)))&
                         (da_model_standard.lat <= np.max(np.array(tp_lat_region)))
                         ,drop=True)


import xesmf as xe

# Regridding to the tracer points
regridder_mlist = {}
for nmodel,model in enumerate(Model_name):
    regridder_list = {}
    for nvar,var in enumerate(Model_varname):
        if (model in [Model_standard]):
            print('model variable same as standard model variable')
        else:
            regridder = xe.Regridder(mean_mlist[model][var],
                                     da_model_standard,
                                     'bilinear',
                                     filename='%s_%s2%s_%s.nc'%(model,var,Model_standard,Variable_standard),
                                     periodic=True,
                                     reuse_weights=False)
            regridder_list['%s_%s2%s_%s'%(model,var,Model_standard,Variable_standard)] = regridder
    regridder_mlist[model] = regridder_list


#### regridding mean field
for nmodel,model in enumerate(Model_name):
    for nvar,var in enumerate(Model_varname):
        if (model in [Model_standard]):
            print('model variable same as standard model variable')
        else:
            mean_mlist[model][var] =               regridder_mlist[model]['%s_%s2%s_%s'%(model,var,Model_standard,Variable_standard)](mean_mlist[model][var])
            mean_mlist[model][var]['x'] = da_model_standard.x.values
            mean_mlist[model][var]['y'] = da_model_standard.y.values




#############################################################################
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig=plt.figure(2,figsize=(20,10))
devy = 0.7

for nmodel,model in enumerate(Model_name):
    #### plotting
    level=np.linspace(-200, 200, 21)
    ax2=fig.add_axes([0,0-devy*nmodel,1,0.5],projection=ccrs.PlateCarree(central_longitude=180))
    im=(mean_mlist[model]['net_heat_coupler']*mean_mlist[model]['net_heat_coupler']/mean_mlist[model]['net_heat_coupler'])\
                 .plot.contourf(x='lon',y='lat',
                                ax=ax2, levels=level,
                                extend='both', cmap='RdBu_r',
                                transform=ccrs.PlateCarree(central_longitude=0.))
    cb=im.colorbar
    cb.remove()
    if nmodel == len(Model_name)-1 :
        cbaxes=fig.add_axes([0.05,0-devy*nmodel-0.1,0.7,0.04])
        cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
        cbar.set_ticks(level)
        cbar.set_ticklabels(["%0.0f"%(n) for n in level]) #
        cbar.ax.tick_params(labelsize=24,rotation=45)
        cbar.set_label(label='W/m$^{2}$',size=24, labelpad=5)
    ax2.coastlines(resolution='110m',linewidths=0.8)
    ax2.add_feature(cfeature.LAND,color='lightgrey')


    ax2.set_xticks([ 60,120,180,240,300], crs=ccrs.PlateCarree())
    ax2.set_xticklabels([ 60,120,180,-120,-60], color='black', weight='bold',size=22)
    ax2.set_yticks([-30,-20,-10,0,10,20,30], crs=ccrs.PlateCarree())
    ax2.set_yticklabels([-30,-20,-10,0,10,20,30], color='black', weight='bold',size=22)
    ax2.yaxis.tick_left()

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_title('%s'%(Model_legend_name[nmodel]), color='black', weight='bold',size=22)
    ax2.set_aspect('auto')
    ax2=None


#### plotting
level=np.linspace(-20, 20, 21)
ax2=fig.add_axes([0,0-devy*(nmodel+1)-0.1,1,0.5],projection=ccrs.PlateCarree(central_longitude=180))
# ax2.set_extent([-180,180,-60,60],crs=ccrs.PlateCarree())
im=(mean_mlist['CORE']['net_heat_coupler']-mean_mlist['JRA']['net_heat_coupler'])\
             .plot.contourf(x='lon',
                            y='lat',
                            ax=ax2,
                            levels=level,
                            extend='both',
                            cmap='RdBu_r',
                            transform=ccrs.PlateCarree(central_longitude=0))

cb=im.colorbar
cb.remove()
if nmodel == len(Model_name)-1 :
    cbaxes=fig.add_axes([0.05,0-devy*(nmodel+1)-0.1-0.1,0.7,0.04])
    cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
    cbar.set_ticks(level)
    cbar.set_ticklabels(["%0.0f"%(n) for n in level])
    cbar.ax.tick_params(labelsize=24,rotation=0)
    cbar.set_label(label='W/m$^{2}$',size=24, labelpad=5)
ax2.coastlines(resolution='110m',linewidths=0.8)
ax2.add_feature(cfeature.LAND,color='lightgrey')

ax2.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
ax2.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
ax2.set_yticks([-30,-20,-10,0,10,20,30], crs=ccrs.PlateCarree())
ax2.set_yticklabels([-30,-20,-10,0,10,20,30], color='black', weight='bold',size=22)
ax2.yaxis.tick_left()

lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.grid(linewidth=2, color='black', alpha=0.3, linestyle='--')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('CORE minus JRA55-do', color='black', weight='bold',size=22)
ax2.set_aspect('auto')

fig.savefig('./figure/figure07.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)
