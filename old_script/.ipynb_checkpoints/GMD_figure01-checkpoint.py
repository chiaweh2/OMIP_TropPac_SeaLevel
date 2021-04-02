#!/usr/bin/env python
# coding: utf-8
import sys
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

"""
The script generate the sea level time series comparison for
eastern tropical Pacific and western tropical Pacific

The longitude range and latitude range can be modified as needed
Currently, it is based on the Western box and Eastern box defined
in the paper.

input files
============
EN.4.2.1.f.analysis.g10.ssl5000  :  the steric sea level from EN4 data (full depth)
                                       (exec io_en4_ssl.py to get this file)
EN.4.2.1.f.analysis.g10.ssl400   :  the steric sea level from EN4 data (400m)
                                       (exec io_en4_ssl.py to get this file)
EN.4.2.1.f.analysis.g10.dtssl400 :  the thermosteric sea level frmo EN4 (400m)
                                       (exec io_en4_tssl.py to get this file)
dt_global_allsat_phy_l4_monthly_adt : the absolute dynamic sea level from CMEMS
                                       (exec io_cmems_sl.py to get this file)

function used
==================
create_ocean_mask.levitus98 : which generate the Pacific basin mask
spherical_area.cal_area     : generate area array based on the lon lat of data


"""



from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=16, processes=False)

from mem_track import used_memory
used_memory()

import warnings
warnings.simplefilter("ignore")




# # ssl 5000
##############################################################################
#### possible input info from external text file
# constant setting
syear = 1993
fyear = 2017
tp_lat_region = [-30,30]     # extract model till latitude
depthint = 5000
# depthint = 400

# regional mean range
lon_range_list = [[120,180],[180,-60],[120,-60]]    # Lon: -180-180
lat_range_list = [[-20,20],[-20,20],[-20,20]]

# inputs
datadir = './data/EN4/'
datafile = 'EN.4.2.1.f.analysis.g10.ssl'
varname = 'ssl'



################################################################
# # Read dataset
#
# read in as dask array to avoid memory overload
ds_data_list = {}
mean_list = {}
season_list = {}

# concat data into one dataset
for yy in range(syear,fyear+1):
    for mm in range(1,13):
        ds0=xr.open_dataset(datadir+'%s%i.%0.4i%0.2i.nc'%(datafile,depthint,yy,mm))
        ds0=ds0.rename(name_dict={'__xarray_dataarray_variable__':varname})
        da0=ds0[varname]
        if yy == syear and mm == 1 :
            da=da0.copy()
        else:
            da=xr.concat([da,da0],dim='time')

# mask land area
da = da.where(da>1E-13,other=np.nan)

# crop data (lat)
da = da.where((da.lat >= np.min(np.array(tp_lat_region)))&
              (da.lat <= np.max(np.array(tp_lat_region)))
              ,drop=True)

# store all model data
ds_data_list['dep%i'%(depthint)] = da


# read mask dataset
ds_pac = levitus98(ds_data_list['dep%i'%(depthint)],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='lon',lat_name='lat', new_regridder_name='')


from spherical_area import cal_area
def area(da_lon,da_lat):
    da_area=xr.apply_ufunc(cal_area,
                           da_lon,da_lat,1.0,1.0,
                           input_core_dims=[[],[],[],[]],
                           vectorize=True,
                           output_dtypes=[np.dtype(np.float64)],
                           dask='parallelized')
    return da_area


# initialize dictionary  (exec this cell will remove all previous calculated values for all variables)
regional_var_list = {}

for nn in range(len(lon_range_list)):
    print('process',lon_range_list[nn],lat_range_list[nn])

    #### setting individual event year range
    lon_range  = lon_range_list[nn]
    lat_range  = lat_range_list[nn]

    # correct the lon range
    lon_range_mod = np.array(lon_range)
    ind1 = np.where(lon_range_mod<np.float(0.))[0]
    lon_range_mod[ind1] = lon_range_mod[ind1]+360. # change Lon range to 0-360

    # crop region
    ds_mask = ds_pac.where(
                          (ds_pac.lon>=np.min(lon_range_mod))&
                          (ds_pac.lon<=np.max(lon_range_mod))&
                          (ds_pac.lat>=np.min(lat_range))&
                          (ds_pac.lat<=np.max(lat_range))
                           ).compute()

    # calculate area
    da_area=area(da.lon,da.lat)*ds_mask        # m^2


    # calculate regional mean
    regional_var_list['%s_dep%i_%i_%i_%i_%i'%(varname,depthint,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]    = ((ds_data_list['dep%i'%(depthint)]*ds_mask*da_area).sum(dim=['lon','lat'])/(ds_mask*da_area).sum(dim=['lon','lat'])).compute()


# # ssl 400
##############################################################################
#### possible input info from external text file
# constant setting
syear = 1993
fyear = 2017
tp_lat_region = [-30,30]     # extract model till latitude
depthint = 400

# regional mean range
lon_range_list = [[120,180],[180,-60],[120,-60]]    # Lon: -180-180
lat_range_list = [[-20,20],[-20,20],[-20,20]]

# inputs
datadir = './data/EN4/'
datafile = 'EN.4.2.1.f.analysis.g10.ssl'
varname = 'ssl'



################################################################
# # Read dataset
#
# read in as dask array to avoid memory overload
ds_data_list = {}
mean_list = {}
season_list = {}

# concat data into one dataset
for yy in range(syear,fyear+1):
    for mm in range(1,13):
        ds0=xr.open_dataset(datadir+'%s%i.%0.4i%0.2i.nc'%(datafile,depthint,yy,mm))
        ds0=ds0.rename(name_dict={'__xarray_dataarray_variable__':varname})
        da0=ds0[varname]
        if yy == syear and mm == 1 :
            da=da0.copy()
        else:
            da=xr.concat([da,da0],dim='time')

# mask land area
da = da.where(da>1E-13,other=np.nan)

# crop data (lat)
da = da.where((da.lat >= np.min(np.array(tp_lat_region)))&
              (da.lat <= np.max(np.array(tp_lat_region)))
              ,drop=True)

# store all model data
ds_data_list['dep%i'%(depthint)] = da

# read mask dataset
ds_pac = levitus98(ds_data_list['dep%i'%(depthint)],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='lon',lat_name='lat', new_regridder_name='')


for nn in range(len(lon_range_list)):
    print('process',lon_range_list[nn],lat_range_list[nn])

    #### setting individual event year range
    lon_range  = lon_range_list[nn]
    lat_range  = lat_range_list[nn]

    # correct the lon range
    lon_range_mod = np.array(lon_range)
    ind1 = np.where(lon_range_mod<np.float(0.))[0]
    lon_range_mod[ind1] = lon_range_mod[ind1]+360. # change Lon range to 0-360

    # crop region
    ds_mask = ds_pac.where(
                          (ds_pac.lon>=np.min(lon_range_mod))&
                          (ds_pac.lon<=np.max(lon_range_mod))&
                          (ds_pac.lat>=np.min(lat_range))&
                          (ds_pac.lat<=np.max(lat_range))
                           ).compute()

    # calculate area
    da_area=area(da.lon,da.lat)*ds_mask        # m^2


    # calculate regional mean
    regional_var_list['%s_dep%i_%i_%i_%i_%i'%(varname,depthint,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]    = ((ds_data_list['dep%i'%(depthint)]*ds_mask*da_area).sum(dim=['lon','lat'])/(ds_mask*da_area).sum(dim=['lon','lat'])).compute()

# # tssl 400
##############################################################################
#### possible input info from external text file
# constant setting
syear = 1993
fyear = 2017
tp_lat_region = [-30,30]     # extract model till latitude
depthint = 400

# regional mean range
lon_range_list = [[120,180],[180,-60],[120,-60]]    # Lon: -180-180
lat_range_list = [[-20,20],[-20,20],[-20,20]]

# inputs
datadir = './data/EN4/'
datafile = 'EN.4.2.1.f.analysis.g10.dtssl'
varname = 'tssl'


################################################################
# # Read dataset
#
# read in as dask array to avoid memory overload
ds_data_list = {}
mean_list = {}
season_list = {}

# concat data into one dataset
for yy in range(syear,fyear+1):
    for mm in range(1,13):
        if yy == syear and mm == 1:
            print("no tssl first month")
        else:
            ds0=xr.open_dataset(datadir+'%s%i_test.%0.4i%0.2i.nc'%(datafile,depthint,yy,mm))
            ds0=ds0.rename(name_dict={'__xarray_dataarray_variable__':varname})
            da0=ds0[varname]
            if yy == syear and mm == 2 :
                da=da0.copy()
            else:
                da=xr.concat([da,da0],dim='time')

# crop data (lat)
da = da.where((da.lat >= np.min(np.array(tp_lat_region)))&
              (da.lat <= np.max(np.array(tp_lat_region)))
              ,drop=True)

# store all model data
ds_data_list['dep%i'%(depthint)] = da

# read mask dataset
ds_pac = levitus98(ds_data_list['dep%i'%(depthint)],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='lon',lat_name='lat', new_regridder_name='')

for nn in range(len(lon_range_list)):
    print('process',lon_range_list[nn],lat_range_list[nn])

    #### setting individual event year range
    lon_range  = lon_range_list[nn]
    lat_range  = lat_range_list[nn]

    # correct the lon range
    lon_range_mod = np.array(lon_range)
    ind1 = np.where(lon_range_mod<np.float(0.))[0]
    lon_range_mod[ind1] = lon_range_mod[ind1]+360. # change Lon range to 0-360

    # crop region
    ds_mask = ds_pac.where(                          (ds_pac.lon>=np.min(lon_range_mod))&                          (ds_pac.lon<=np.max(lon_range_mod))&                          (ds_pac.lat>=np.min(lat_range))&                          (ds_pac.lat<=np.max(lat_range))                           ).compute()

    # calculate area
    da_area=area(da.lon,da.lat)*ds_mask        # m^2


    # calculate regional mean
    regional_var_list['%s_dep%i_%i_%i_%i_%i'%(varname,depthint,lon_range[0],lon_range[1],lat_range[0],lat_range[1])]    = ((ds_data_list['dep%i'%(depthint)]*ds_mask*da_area).sum(dim=['lon','lat'])/(ds_mask*da_area).sum(dim=['lon','lat'])).compute()


# # CMEMS sl
#### possible input info from external text file
# constant setting
obs_year_range = [[1993,2018,9]]

tp_lat_region = [-30,30]     # extract model till latitude

Obs_varname = [['adt']]

Obs_name = ['CMEMS']

# inputs
obsin = {}
obspath = {}

obs = Obs_name[0]
obsdir = './data/CMEMS/'
obsfile = [['dt_global_allsat_phy_l4_monthly_adt.nc']]
obspath[obs]=[obsdir,obsfile]


for nobs,obs in enumerate(Obs_name):
    obsdir = obspath[obs][0]
    obsfile = obspath[obs][1]
    multivar = []
    for file in obsfile :
        if len(file) == 1 :
            multivar.append([os.path.join(obsdir,file[0])])
        elif len(file) > 1 :
            multifile = []
            for ff in file :
                multifile.append(os.path.join(obsdir,ff))
            multivar.append(multifile)
    obsin[obs] = multivar

# initialization of dict and list  (!!!!!!!! remove all previous read model info if exec !!!!!!!!!!)
ds_obs_mlist = {}
obs_mean_mlist = {}
obs_season_mlist = {}
obs_linear_mlist = {}


import spherical_area as sa

#### obs
for nobs,obs in enumerate(Obs_name):
    ds_obs_list = {}
    obs_mean_list = {}
    obs_season_list = {}
    obs_linear_list = {}
    for nvar,var in enumerate(Obs_varname[nobs]):
        print('read %s %s'%(obs,var))

        # read input data
        #-- single file
        if len(obsin[obs][nvar]) == 1 :

            # find out dimension name
            da = xr.open_dataset(obsin[obs][nvar][0],chunks={})
            obsdims = list(da[var].dims)

            ds_obs = xr.open_dataset(obsin[obs][nvar][0],chunks={obsdims[0]:50,obsdims[1]:50,obsdims[2]:50},use_cftime=True)

        #-- multi-file merge (same variable)
        elif len(obsin[obs][nvar]) > 1 :
            for nf,file in enumerate(obsin[obs][nvar]):
                # find out dimension name
                da = xr.open_dataset(file,chunks={})
                obsdims = list(da[var].dims)

                ds_obs_sub = xr.open_dataset(file,chunks={obsdims[0]:50,obsdims[1]:50,obsdims[2]:50},use_cftime=True)
                if nf == 0 :
                    ds_obs = ds_obs_sub
                else:
                    ds_obs = xr.concat([ds_obs,ds_obs_sub],dim='time',data_vars='minimal')

        ############## CMEMS ##############
        if obs in ['CMEMS']:
            syear_obs = obs_year_range[nobs][0]
            fyear_obs = obs_year_range[nobs][1]
            fmon_obs = obs_year_range[nobs][2]
            #### create time axis for overlapping period
            timeax = xr.cftime_range(start=cftime.datetime(syear_obs,1,1),end=cftime.datetime(fyear_obs,fmon_obs,1),freq='MS')
            timeax = timeax.to_datetimeindex()    # cftime => datetime64
            ds_obs.time.values = timeax

            # calculate global mean sea level
            da_area = sa.da_area(ds_obs, lonname='longitude', latname='latitude',
                                 xname='longitude', yname='latitude', model=None)
            da_glo_mean = (ds_obs*da_area).sum(dim=['longitude','latitude'])/da_area.sum(dim=['longitude','latitude'])
#             ds_obs = ds_obs-da_glo_mean

            # rename
            ds_obs = ds_obs.rename({'longitude':'lon','latitude':'lat'})
        else:
            syear_obs = obs_year_range[nobs][0]
            fyear_obs = obs_year_range[nobs][1]
            #### create time axis for overlapping period
            timeax = xr.cftime_range(start=cftime.datetime(syear_obs,1,1),end=cftime.datetime(fyear_obs,12,31),freq='MS')
            timeax = timeax.to_datetimeindex()    # cftime => datetime64
            ds_obs.time.values = timeax


        # crop data (time)
        ds_obs = ds_obs[var]\
        .where((ds_obs['time.year'] >= syear)&\
               (ds_obs['time.year'] <= fyear)\
               ,drop=True)
        ds_obs = ds_obs\
        .where((ds_obs.lat >= np.min(np.array(tp_lat_region)))&\
               (ds_obs.lat <= np.max(np.array(tp_lat_region)))\
               ,drop=True)

        # store all model data
        ds_obs_list[var] = ds_obs

    ds_obs_mlist[obs] = ds_obs_list

# read mask dataset
ds_pac = levitus98(ds_obs_list[Obs_varname[0][0]],
                    basin=['pac'],reuse_weights=True, newvar=True,
                    lon_name='lon',lat_name='lat', new_regridder_name='')


for nn in range(len(lon_range_list)):
    print('process',lon_range_list[nn],lat_range_list[nn])

    #### setting individual event year range
    lon_range  = lon_range_list[nn]
    lat_range  = lat_range_list[nn]

    # correct the lon range
    lon_range_mod = np.array(lon_range)
    ind1 = np.where(lon_range_mod<np.float(0.))[0]
    lon_range_mod[ind1] = lon_range_mod[ind1]+360. # change Lon range to 0-360

    # crop region
    ds_mask = ds_pac.where(
                          (ds_pac.lon>=np.min(lon_range_mod))&
                          (ds_pac.lon<=np.max(lon_range_mod))&
                          (ds_pac.lat>=np.min(lat_range))&
                          (ds_pac.lat<=np.max(lat_range))
                           ).compute()

    # calculate area
    da_area=area(ds_obs_list[Obs_varname[0][0]].lon,ds_obs_list[Obs_varname[0][0]].lat)*ds_mask        # m^2


    # calculate regional mean
    regional_var_list['%s_%i_%i_%i_%i'%(Obs_varname[0][0],lon_range[0],lon_range[1],lat_range[0],lat_range[1])]  = \
     ((ds_obs_list[Obs_varname[0][0]]*ds_mask*da_area).sum(dim=['lon','lat'])/(ds_mask*da_area).sum(dim=['lon','lat'])).compute()

    regional_var_list['%s_%i_%i_%i_%i'%(Obs_varname[0][0],lon_range[0],lon_range[1],lat_range[0],lat_range[1])]  = \
     regional_var_list['%s_%i_%i_%i_%i'%(Obs_varname[0][0],lon_range[0],lon_range[1],lat_range[0],lat_range[1])]\
      .where(regional_var_list['%s_%i_%i_%i_%i'%(Obs_varname[0][0],lon_range[0],lon_range[1],lat_range[0],lat_range[1])] != 0.
           ,drop=True)


def var_exp(ts1,ts2):

    sumts = ts1+ts2
    ts1 = ts1*sumts/sumts
    ts2 = ts2*sumts/sumts
    slope, intercept, r_value, pval, std_err = stats.linregress(ts1.values,ts2.values)
    return r_value**2



import matplotlib.pyplot as plt
import datetime
from scipy import stats

fig=plt.figure(figsize=(24,10))
ax1=fig.add_axes([0,0,0.8,0.6])

region = '120_180_-20_20'
ssl = regional_var_list['ssl_dep5000_%s'%region]-regional_var_list['ssl_dep5000_%s'%region].mean()
ssl400 = regional_var_list['ssl_dep400_%s'%region]-regional_var_list['ssl_dep400_%s'%region].mean()
tssl400 = (regional_var_list['tssl_dep400_%s'%region]-regional_var_list['tssl_dep400_%s'%region].mean()).cumsum()
sl = regional_var_list['adt_%s'%region]-regional_var_list['adt_%s'%region].mean()
sl['time'] = ssl.time

varexp = var_exp(ssl,sl)
ssl.plot(ax=ax1,label='Steric (EN4)',linewidth=3.0)

varexp = var_exp(ssl400,ssl)
ssl400.plot(ax=ax1,label='Steric upper 400m (EN4)',linewidth=3.0)

varexp = var_exp(tssl400,ssl400)
tssl400.plot(ax=ax1,label='Thermosteric (EN4)',linewidth=3.0)

sl.plot(ax=ax1,label='Total (CMEMS)',linewidth=3.0,color='k')



sllevel=np.linspace(-0.15,0.15,11)

#### setting the plotting format
ax1.set_ylabel('Sea Level (cm)',{'size':'20'})
ax1.tick_params(axis='y',labelsize=20)
# ax1.grid(linestyle='dashed',axis='both')
ax1.set_xlabel('',{'size':'20'})
ax1.set_title("Western Tropical Pacific",{'size':'22'},pad=24)
ax1.tick_params(axis='x',labelsize=20)
ax1.set_yticks(sllevel)
ax1.set_yticklabels(['%2.0f'%tick for tick in sllevel*100.])
# ax1.legend(loc='lower right',fontsize=15,frameon=False)


ax2=fig.add_axes([0,-0.8,0.8,0.6])

region = '180_-60_-20_20'

ssl = regional_var_list['ssl_dep5000_%s'%region]-regional_var_list['ssl_dep5000_%s'%region].mean()
ssl400 = regional_var_list['ssl_dep400_%s'%region]-regional_var_list['ssl_dep400_%s'%region].mean()
tssl400 = (regional_var_list['tssl_dep400_%s'%region]-regional_var_list['tssl_dep400_%s'%region].mean()).cumsum()
sl = regional_var_list['adt_%s'%region]-regional_var_list['adt_%s'%region].mean()

sl['time'] = ssl.time

varexp = var_exp(ssl,sl)
ssl.plot(ax=ax2,label='Steric (EN4)',linewidth=3.0)

varexp = var_exp(ssl400,ssl)
ssl400.plot(ax=ax2,label='Steric upper 400m (EN4)',linewidth=3.0)

varexp = var_exp(tssl400,ssl400)
tssl400.plot(ax=ax2,label='Thermosteric upper 400m (EN4)',linewidth=3.0)

sl.plot(ax=ax2,label='Total (CMEMS)',linewidth=3.0,color='k')




sllevel=np.linspace(-0.15,0.15,11)

#### setting the plotting format
ax2.set_ylabel('Sea Level (cm)',{'size':'20'})
ax2.tick_params(axis='y',labelsize=20)
# ax2.grid(linestyle='dashed',axis='both')
ax2.set_xlabel('Year',{'size':'20'})
ax2.set_title("Eastern Tropical Pacific",{'size':'22'},pad=24)
ax2.tick_params(axis='x',labelsize=20)
ax2.set_yticks(sllevel)
ax2.set_yticklabels(['%2.0f'%tick for tick in sllevel*100.])
ax2.legend(loc='lower right',fontsize=22,frameon=False)


fig.savefig('./figure/figure1.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)
