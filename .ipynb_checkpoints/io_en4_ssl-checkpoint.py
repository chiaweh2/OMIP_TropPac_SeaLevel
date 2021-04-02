#!/usr/bin/env python
# coding: utf-8

# # IO for EN4 data
import zipfile
import xarray as xr
import gsw
import time
import sea_prop
import dask
import os, psutil
import numpy as np

dask.config.set(scheduler='threads',num_workers=8)

### setting
syear=1992
fyear=2020
depthint=400

# # Determine density
# The density is determined based on TEOS10 http://www.teos-10.org/
#
# * Use [python gsw package](https://teos-10.github.io/GSW-Python/intro.html) to calcualte density
# * Available data Theta (potential temperature in degree C) + Salt (salinity in psu)
# * [`gsw.density.rho(SA, CT, p)`](https://teos-10.github.io/GSW-Python/density.html) calculates in-situ density from Absolute Salinity and Conservative Temperature, using the computationally-efficient expression for specific volume in terms of SA, CT and p (Roquet et al., 2015).
#
# ### Steps
# 0. change depth in meter to pressure using `gsw.conversions.p_from_z(z, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0)`
# 1. change salinity from PSU (practical salinity unit/PSS-78) to absolute salinity (mass fraction, in grams per kilogram of solution) using `gsw.conversions.SA_from_SP(SP, p, lon, lat)`
# 2. change potential temperature to conservative temperature using `gsw.conversions.CT_from_pt(SA, pt)`
# 3. calculate $\rho$ using `gsw.density.rho(SA, CT, p)`


@dask.delayed
def z_to_p(da_z,da_lat):
    # input: Depth, positive up in meter
    # output: Pressure dbar
    #        (( i.e. absolute pressure - 10.1325 dbar ))
    da_p=xr.apply_ufunc(gsw.conversions.p_from_z,
                        da_z,da_lat,
                        input_core_dims=[[],[]],
                        vectorize=True,
                        kwargs={'geo_strf_dyn_height':0,
                                'sea_surface_geopotential':0},
                        dask='allowed')
    return da_p

@dask.delayed
def SP_to_SA(da_sp,da_p,da_lon,da_lat):
    # input: Practical Salinity (PSS-78)
    # output: Absolute Salinity (g/kg)
    da_sa=xr.apply_ufunc(gsw.conversions.SA_from_SP,
                        da_sp,da_p,da_lon,da_lat,
                        input_core_dims=[[],[],[],[]],
                        vectorize=True,
                        dask='allowed')
    return da_sa

@dask.delayed
def pt_to_CT(da_sa,da_pt):
    # input: Potential temp (degrees C)
    # output: Conservative Temperature (ITS-90)
    da_ct=xr.apply_ufunc(gsw.conversions.CT_from_pt,
                        da_sa,da_pt,
                        input_core_dims=[[],[]],
                        vectorize=True,
                        dask='allowed')
    return da_ct

@dask.delayed
def teos10_rho(da_sa,da_ct,da_p):
    # input: Absolute Salinity (g/kg)
    #        Conservative Temperature (ITS-90, degrees C)
    #        Sea pressure (absolute pressure minus 10.1325 dbar)
    # output: Density (kg/m) ===> typo on the gsw website (should be kg/m^3)
    da_rho=xr.apply_ufunc(gsw.density.rho,
                        da_sa,da_ct,da_p,
                        input_core_dims=[[],[],[]],
                        vectorize=True,
                        dask='allowed')
    return da_rho

@dask.delayed
def steric_sl(da_rho,da_dp):
    # input: dp  (Pa)
    #        rho (kg/m^3)
    # output: H (m)

    da_ssl=xr.apply_ufunc(sea_prop.steric_hgt,
                        da_rho,da_dp,
                        input_core_dims=[['depth'],['depth']],
                        vectorize=True,
                        dask='allowed')
    return da_ssl




en4dir='../data/EN4/'
zipname=['EN.4.2.1.analyses.g10.%0.4i.zip'%(year) for year in range(syear,fyear)]



for file in zipname:
    archive=zipfile.ZipFile(en4dir+file, 'r')          # open one zip file
    filenames=archive.namelist()                       # read the files' name in the zip
    for monfile in filenames:
        ncfile=archive.open(monfile)                   # extract one file in the zip
        print("==============================")
        print("processing %s"%(ncfile.name))
        ds=xr.open_dataset(ncfile,chunks={'lon':50,'lat':50,'depth':42})

        # remove the level below certain depth
        ds=ds.where(ds.depth<=depthint,drop=True)

        # output memory used so far
        p = psutil.Process()
        print(p.memory_info().rss/1024**2,'MB')

        # reciept of calculating steric sea level
        da_p=z_to_p(-ds.depth, ds.lat)                                # dbar
        da_sa=SP_to_SA(ds.salinity.isel(time=0),da_p,ds.lon,ds.lat)   # g/kg
        da_ct=pt_to_CT(da_sa,ds.temperature.isel(time=0)-273.15)      # deg C
        da_rho=teos10_rho(da_sa,da_ct,da_p)                           # kg/m^3
        da_p1=z_to_p(-ds.depth_bnds[:,0], ds.lat)                     # dbar
        da_p2=z_to_p(-ds.depth_bnds[:,1], ds.lat)                     # dbar
        da_dp=np.abs(da_p1-da_p2)*1e4                                 # Pa (1dbar=1x10^4Pa)
        da_ssl=steric_sl(da_rho,da_dp)                                # m

        # start time
        timestart=time.process_time()

        # compute the dask array (Parallized)
        da_ssl_temp=da_ssl.compute()
        da_ssl_temp.to_netcdf(en4dir+ncfile.name[:-9]+'ssl%i.'%depthint+ncfile.name[-9:])

        # end time
        timelapse=time.process_time()-timestart
        print(timelapse/60.,'mins')

        # output memory used so far
        p = psutil.Process()
        print(p.memory_info().rss/1024**2,'MB')
