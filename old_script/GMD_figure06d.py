#!/usr/bin/env python
# coding: utf-8

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
The script generate the vertical velocity transect along 180
in the form of
figure 6d : vertical transect along 180
in the paper.

input files
============
JRA55-do : uo


function used
==================
plotting_func.shiftedColorMap : shift the colorbar based on zero value


"""

from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=4, processes=False)

import warnings
warnings.simplefilter("ignore")

# # Read N-S transect datasets
# define time range
time_range = [1993,2007]
transect_str = '180'

# read transect ncfiles
modeldir = './data/GFDL/JRA/transect_scpt/'
modelfile = 'JRA_uo_thetao_yz_-180/'
ds_transect = xr.open_mfdataset(os.path.join(modeldir,modelfile,'????-??.nc'),concat_dim='time').load()
da_uo = ds_transect['uo']\
                  .where((ds_transect['time.year'] >= np.min(time_range))&
                         (ds_transect['time.year'] <= np.max(time_range))
                         ,drop=True).squeeze()

da_uo_mean = da_uo.mean(dim='time')

da_thetao = ds_transect['thetao']\
                  .where((ds_transect['time.year'] >= np.min(time_range))&
                         (ds_transect['time.year'] <= np.max(time_range))
                         ,drop=True).squeeze()

da_thetao_mean = da_thetao.mean(dim='time')

# # Plot mean `uo`

from plotting_func import shiftedColorMap
import matplotlib


# climatology of uo
#### setting individual event year range
lat_range  = [-35,35]
lat_label  = [-30,-20,-10,0,10,20,30]
dep_range  = [0,700]
tlevel     = np.linspace(-0.2, 0.7, 19)
model      = 'JRA'

#### plotting
fig = plt.figure(2,figsize=(20,10))

ax2 = fig.add_axes([0,0,1.25,0.6])

cmap = matplotlib.cm.RdBu_r
midind = int(len(tlevel) / 2)
cmap = shiftedColorMap(cmap, midpoint=tlevel[midind], name='shifted')

im = da_uo_mean.squeeze()\
        .where((da_uo_mean.y>=lat_range[0])&
               (da_uo_mean.y<=lat_range[1])&
               (da_uo_mean.z>=dep_range[0])&
               (da_uo_mean.z<=dep_range[1]),drop=True)\
        .plot.contourf(x='y',
                       y='z',
                       ax=ax2,
                       levels=tlevel,
                       cmap=cmap,
                       extend='both')

ax2.set_ylim(dep_range)
ax2.invert_yaxis()
ax2.set_ylabel('Depth (m)',{'size':'20'})
ax2.tick_params(axis='y',labelsize=20)
# ax2.set_yticks(depth)
# ax2.set_yticklabels(depth)

ax2.set_xlim(lat_range)
ax2.set_xlabel('Latitude',{'size':'20'})
ax2.tick_params(axis='x',labelsize=20)
ax2.set_xticks(lat_label)
# ax2.set_xticklabels(lat_label)
ax2.set_xticklabels(['30$^\circ$S','20$^\circ$S','10$^\circ$S','0$^\circ$','10$^\circ$N','20$^\circ$N','30$^\circ$N'], color='black',size=22)


cb=im.colorbar
cb.remove()
cbaxes=fig.add_axes([0,-0.13,1, 0.03])
cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
cbar.set_ticks(tlevel)
cbar.set_ticklabels(["%0.2f"%(n) for n in tlevel])
# cbar.set_ticklabels([" " for n in tlevel])
cbar.ax.tick_params(labelsize=22)
cbar.set_label(label='Zonal Current (m/s)',size=18, labelpad=15)


ax2.set_title("Pacific transect at %s$^\circ$ (1993-2007 Climatology)"%transect_str,{'size':'22'},pad=24)
ax2.grid(linestyle='dashed',alpha=0.2,color='k')

fig.savefig('./figure/figure06d.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)
