
#!/usr/bin/env python


# # IO for GFDL Ocean model MOM6 OMIP outputs
#     The script starts from reading origianl data output from GFDL models.
#     To increase the speed of processing including IO and dask chunking,
#     zarr format is created
# to reduce some data storage the data is always cropped to 50S-50N
# the output is for tropical pacific analysis so should be enough

# needed python module dependence
# 1) cftime
# 2) dask
# 3) os
# 4) xarray
# 5) numpy


import os
import sys
import warnings
import cftime
import dask
import xarray as xr
import numpy as np


#### call distributed scheduler
warnings.simplefilter('ignore')

from dask.distributed import Client, LocalCluster
cluster = LocalCluster(processes=False,n_workers=1,threads_per_worker=8)
client = Client(cluster)

# set the output latitude range
lat_range = [-90,90]
# lat_range = [-50,50]  # decrease zarr file storage space




# #####################################################

# #### possible input info from external text file
# # input
# syear = 1948
# fyear = 2007

# model = 'CORE'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/CORE/OM4p25_IAF_BLING_csf_rerun_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['ocean_monthly.194801-200712.tos.nc']

# Model_varname = ['tos']

# Coord_name = [['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat']]

# Area_name = ['areacello']

# # output location info

# # the extended filename added to the end of the name
# ext = ['']     

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'../data/')

# #####################################################

# #### possible input info from external text file
# # input
# syear = 1948
# fyear = 2007

# model = 'CORE'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/CORE/OM4p25_IAF_BLING_csf_rerun_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['ocean_monthly.194801-200712.hflso.nc',
#              'ocean_monthly.194801-200712.hfsso.nc',
#              'ocean_monthly.194801-200712.rlntds.nc',
#              'ocean_monthly.194801-200712.rsntds.nc']

# Model_varname = ['hflso',
#                  'hfsso',
#                  'rlntds',
#                  'rsntds']

# Coord_name = [['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat']]

# Area_name = ['areacello',
#              'areacello',
#              'areacello',
#              'areacello']

# # output location info

# # the extended filename added to the end of the name
# ext = ['',
#        '',
#        '',
#        '']        

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'../data/')


# #####################################################

# #### possible input info from external text file
# # input
# syear = 1948
# fyear = 1967

# model = 'CORE'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/CORE/OM4p25_IAF_BLING_csf_rerun_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['ocean_monthly_z.194801-196712.thetao.k1-16.nc',
#              'ocean_monthly_z.194801-196712.so.k1-16.nc',
#              'ocean_monthly_z.194801-196712.uo.k1-16.nc',
#              'ocean_monthly_z.194801-196712.vo.k1-16.nc']

# Model_varname = ['thetao',
#                  'so',
#                  'uo',
#                  'vo']

# Coord_name = [['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon_u','geolat_u'],
#               ['geolon_v','geolat_v']]

# Area_name = ['areacello',
#              'areacello',
#              'areacello_cu',
#              'areacello_cv']

# # output location info

# # the extended filename added to the end of the name
# ext = ['_1948_1967',
#        '_1948_1967',
#        '_1948_1967',
#        '_1948_1967']        

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'../data/')


# ######################################################
# # #### possible input info from external text file
# # !!!!
# # output dir need to be update in the code to avoid zarr file rewrite
# # !!!!
# # # input
# syear = 1948
# fyear = 2007

# model = 'CORE'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/CORE/OM4p25_IAF_BLING_csf_rerun_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['ocean_monthly.194801-200712.zos.nc']

# Model_varname = ['zos']

# Coord_name = [['geolon','geolat']]

# Area_name = ['areacello']

# # output location info

# # the extended filename added to the end of the name
# ext = ['_all']        

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'../data/')

# ######################################################
# #### possible input info from external text file
#!!!!
# output dir need to be update in the code to avoid zarr file rewrite
#!!!!
# # input
syear = 1958
fyear = 2017

model = 'JRA'

modeldir = '/storage2/chiaweih/OMIP/GFDL/JRA/OM4p25_JRA55do1.4_0netfw_cycle6/'

gridfile = 'ocean_monthly.static.nc'

modelfile = ['ocean_monthly.195801-201712.zos.nc']

Model_varname = ['zos']

Coord_name = [['geolon','geolat']]

Area_name = ['areacello']

# output location info

# the extended filename added to the end of the name
ext = ['_all']        

basedir = os.getcwd()
outputdir = os.path.join(basedir,'../data/')

######################################################

# #### possible input info from external text file
# # input
# syear = 1968
# fyear = 1992

# model = 'CORE'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/CORE/OM4p25_IAF_BLING_csf_rerun_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['ocean_monthly_z.196801-199212.thetao.k1-16.nc',
#              'ocean_monthly_z.196801-199212.uo.k1-16.nc',
#              'ocean_monthly_z.196801-199212.vo.k1-16.nc'
#             ]

# Model_varname = ['thetao',
#                  'uo',
#                  'vo']

# Coord_name = [['geolon','geolat'],
#               ['geolon_u','geolat_u'],
#               ['geolon_v','geolat_v']
#               ]

# Area_name = ['areacello',
#              'areacello_cu',
#              'areacello_cv'
#              ]

# # output location info

# # the extended filename added to the end of the name
# ext = ['_1968_1992',
#        '_1968_1992',
#        '_1968_1992']        

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'../data/')


######################################################

# #### possible input info from external text file
# # input
# syear = 1958
# fyear = 2017

# model = 'JRA'

# modeldir = '/storage2/chiaweih/OMIP/GFDL/JRA/OM4p25_JRA55do1.4_0netfw_cycle6/'

# gridfile = 'ocean_monthly.static.nc'

# modelfile = ['Pacific.195801-201712.hflso.nc',
#              'Pacific.195801-201712.hfsso.nc',
#              'Pacific.195801-201712.rlntds.nc',
#              'Pacific.195801-201712.rsntds.nc'
#             ]

# Model_varname = ['hflso',
#                  'hfsso',
#                  'rlntds',
#                  'rsntds']

# Coord_name = [['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat'],
#               ['geolon','geolat']
#               ]

# Area_name = ['areacello',
#              'areacello',
#              'areacello',
#              'areacello'
#              ]

# # output location info

# # the extended filename added to the end of the name
# ext = ['',
#        '',
#        '',
#        '']        

# basedir = os.getcwd()
# outputdir = os.path.join(basedir,'')



#############################################################################
# initialize input file dict
modelin_mlist = {}
modelgd_mlist = {}
Model_name = []

# list of file paths
modelin = [os.path.join(modeldir,file) for file in modelfile]
modelgd = os.path.join(modeldir,gridfile)
modelin_mlist[model] = modelin
modelgd_mlist[model] = modelgd
Model_name.append(model)


#### models
for nmodel, model in enumerate(Model_name):
    for nvar, var in enumerate(Model_varname):
        print('calculate model data')
        print(modelin_mlist[model][nvar])

        # find out dimension name
        #   usually dimension is in the order of
        #   => time, depth, lat, lon
        da = xr.open_dataset(modelin_mlist[model][nvar],chunks={})
        modeldims = list(da[var].dims)
        chunks={}
        for dim in modeldims:
            chunks[dim]=50

        # read input data
        ds_model = xr.open_dataset(modelin_mlist[model][nvar],chunks=chunks,use_cftime=True)
        ds_grid = xr.open_dataset(modelgd_mlist[model],chunks={})

        # temp solution to correct the v lon at pole !!!!!!!!!!!!!!!!!!!!!!!!!!
        # there are some weird gridding in high latitude for geolon_v
        # only for MOM6 only
        ds_grid.geolon_v.values[-1,:]=ds_grid.geolon.values[-1,:]

        # crop grid to model size (for data that is not global)
        ds_grid = ds_grid.where((ds_grid[modeldims[-2]] >= np.min(ds_model[modeldims[-2]]))&
                                (ds_grid[modeldims[-2]] <= np.max(ds_model[modeldims[-2]]))&
                                (ds_grid[modeldims[-1]] >= np.min(ds_model[modeldims[-1]]))&
                                (ds_grid[modeldims[-1]] <= np.max(ds_model[modeldims[-1]]))
                                ,drop=True)

        # create new dataset structure (design for both 3d and 2d)
        #   coord is renamed to lon lat
        #   the new dimension order is time, depth, yi, xi
        #   (i depend on the arakawa grid)
        if len(modeldims) == 4:
            ds_model_merge = xr.Dataset(coords={
                                            'lon':((modeldims[-2],modeldims[-1]),ds_grid[Coord_name[nvar][0]].values),
                                            'lat':((modeldims[-2],modeldims[-1]),ds_grid[Coord_name[nvar][1]].values),
                                            modeldims[-4]:ds_model[modeldims[-4]].values,
                                            modeldims[-3]:ds_model[modeldims[-3]].values,
                                            modeldims[-2]:ds_model[modeldims[-2]].values,
                                            modeldims[-1]:ds_model[modeldims[-1]].values,})

        elif len(modeldims) == 3:
            ds_model_merge = xr.Dataset(coords={
                                            'lon':((modeldims[-2],modeldims[-1]),ds_grid[Coord_name[nvar][0]].values),
                                            'lat':((modeldims[-2],modeldims[-1]),ds_grid[Coord_name[nvar][1]].values),
                                            modeldims[-3]:ds_model[modeldims[-3]].values,
                                            modeldims[-2]:ds_model[modeldims[-2]].values,
                                            modeldims[-1]:ds_model[modeldims[-1]].values,})

        ds_model_merge[var] = ds_model[var]

        # crop data (temporal)
        ds_model_merge = ds_model_merge.where(\
                            (ds_model_merge['time.year'] >=syear)&\
                            (ds_model_merge['time.year'] <=fyear),drop=True)

        # add areacello
        ds_model_merge[Area_name[nvar]] = ds_grid[Area_name[nvar]]

        # rename dim
        if len(modeldims) == 4:
            ds_model_merge = ds_model_merge.rename({modeldims[-3]:'z',modeldims[-2]:'y',modeldims[-1]:'x'})
            ds_model_merge = ds_model_merge.chunk(chunks={'time':-1,'z':-1,'y':100,'x':100})
        elif len(modeldims) == 3:
            ds_model_merge = ds_model_merge.rename({modeldims[-2]:'y',modeldims[-1]:'x'})
            ds_model_merge = ds_model_merge.chunk(chunks={'time':-1,'y':100,'x':100})

        # change the cftime to np.datetime for easy plotting boundary setting
        if len(ds_model_merge['time']) > 1:
            timeax = xr.cftime_range(start=cftime.datetime(ds_model_merge['time.year'][0],1,1),
                                     end=cftime.datetime(ds_model_merge['time.year'][-1],12,31),
                                     freq='MS')
            timeax = timeax.to_datetimeindex()    # cftime => datetime64
            ds_model_merge['time'] = timeax


        # crop data (spatial)
        ds_model_merge = ds_model_merge.where((ds_model_merge.lat >=np.min(lat_range))&\
                                              (ds_model_merge.lat <=np.max(lat_range)),\
                                              drop=True).persist()

        # rechunking data
        if len(modeldims) == 4:
            ds_model_merge=ds_model_merge.chunk(chunks={'time':-1,'z':-1,'y':100,'x':100})
        elif len(modeldims) == 3:
            ds_model_merge=ds_model_merge.chunk(chunks={'time':-1,'y':100,'x':100})

        # store all model data
        ds_model_merge.to_zarr(outputdir+'GFDL/%s/%s_%s%s.zarr'%(model,model,var,ext[nvar]),mode='w')
