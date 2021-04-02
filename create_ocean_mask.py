import xesmf as xe
import os
import xarray as xr


def levitus98(da_var,basin=['all'],reuse_weights=True, newvar=False, lon_name='x',lat_name='y', new_regridder_name=''):
    """
    The function is designed to create ocean mask that is larger than the data 
    region. Due to this design, the mask can include all available points in 
    the dataset and not accidentally cropping out the data that exist in the 
    data in the basin. The ocean basin mask is based on the Levitus 1998 ncfiles
    
    Caution!!!
    - make sure the data is set to NaN on land. 
    
    
    input :
    
        da_var(xr.DataArray) - the var one want the ocean basin to applied on.
        The var is needed for its grid points not its value in this function.
        
    Parameters :        
    
        basin(list with only one string) - The list of ocean basin one want in the output. 
        Default is to output all 3 ocean basins, the sum of all three basins, and indopacific.
        To output only one basin => ['pacific'],['atlantic'], or ['indian']
        
        reuse_weights(Boolean) - set to False to erase the weight file after finished 
        regridding. Default is to save the regridder file.  
        
        newvar(Boolean) - set to True if one want to create new regridder/file. Default is False 
        to reuse the previous regridder.
        
    Returns:
        
        da_mask (multiple/single xr.DataArray) - the total output number of mask depending 
        on the kwarg basin.
            
    
    """

    if newvar:
        try :
            os.remove('basin_pacific_regrid%s.nc'%(new_regridder_name))
        except FileNotFoundError:
            print('No previous Pacific regridder file')
            
        try :
            os.remove('basin_atlantic_regrid%s.nc'%(new_regridder_name))
        except FileNotFoundError:
            print('No previous Atlantic regridder file')
            
        try :
            os.remove('basin_indian_regrid%s.nc'%(new_regridder_name))
        except FileNotFoundError:
            print('No previous Indian regridder file')
            



    input_file = '/storage1/home1/chiaweih/Research/proj3_omip_sl/data/Levitus1998/'

    if 'pac' in basin or 'all' in basin:
        ncfile = 'pacific.nc'
        ds_mask_pacific = xr.open_dataset(input_file+ncfile)

#         ds_mask_pacific.plot()
        # Regridding to the tracer points
        regridder_mask = xe.Regridder(ds_mask_pacific,\
                                      da_var,\
                                      'bilinear',
                                      filename='basin_pacific_regrid%s.nc'%(new_regridder_name),
                                      periodic=True,
                                      reuse_weights=reuse_weights)
        da_mask_pacific_regrid = regridder_mask(ds_mask_pacific.z)
        da_mask_pacific_regrid[lon_name] = da_var[lon_name]
        da_mask_pacific_regrid[lat_name] = da_var[lat_name]
        
        if reuse_weights is False:
            regridder_mask.clean_weight_file()
 

    if 'ind' in basin or 'all' in basin:
        ncfile = 'indian.nc'
        ds_mask_indian = xr.open_dataset(input_file+ncfile)


        # Regridding to the tracer points
        regridder_mask = xe.Regridder(ds_mask_indian,\
                                      da_var,\
                                      'bilinear',
                                      filename='basin_indian_regrid%s.nc'%(new_regridder_name),
                                      periodic=True,
                                      reuse_weights=reuse_weights)
        da_mask_indian_regrid = regridder_mask(ds_mask_indian.z)
        da_mask_indian_regrid[lon_name] = da_var[lon_name]
        da_mask_indian_regrid[lat_name] = da_var[lat_name]
        if reuse_weights is False:
            regridder_mask.clean_weight_file()
 


    if 'atl' in basin or 'all' in basin:
        ncfile = 'atlantic.nc'
        ds_mask_atl = xr.open_dataset(input_file+ncfile)


        # Regridding to the tracer points
        regridder_mask = xe.Regridder(ds_mask_atl,\
                                      da_var,\
                                      'bilinear',
                                      filename='basin_atlantic_regrid%s.nc'%(new_regridder_name),
                                      periodic=True,
                                      reuse_weights=reuse_weights)
        da_mask_atlantic_regrid = regridder_mask(ds_mask_atl.z)
        da_mask_atlantic_regrid[lon_name] = da_var[lon_name]
        da_mask_atlantic_regrid[lat_name] = da_var[lat_name]
        if reuse_weights is False:
            regridder_mask.clean_weight_file()
 
    if 'all' in basin:
        da_indopac = da_mask_indian_regrid+da_mask_pacific_regrid
        da_3basin = da_mask_atlantic_regrid+da_mask_indian_regrid+da_mask_pacific_regrid
    
    if 'atl' in basin:
        da_mask_atlantic_regrid = da_mask_atlantic_regrid.where(da_mask_atlantic_regrid>0.)
        da_mask_atlantic_regrid = da_mask_atlantic_regrid.where(da_mask_atlantic_regrid.isnull(),other=1.)
        return da_mask_atlantic_regrid
    
    if 'ind' in basin:
        da_mask_indian_regrid = da_mask_indian_regrid.where(da_mask_indian_regrid>0.)
        da_mask_indian_regrid = da_mask_indian_regrid.where(da_mask_indian_regrid.isnull(),other=1.)
        return da_mask_indian_regrid
    
    if 'pac' in basin:
        da_mask_pacific_regrid = da_mask_pacific_regrid.where(da_mask_pacific_regrid>0.)
        da_mask_pacific_regrid = da_mask_pacific_regrid.where(da_mask_pacific_regrid.isnull(),other=1.)
        return da_mask_pacific_regrid
    
    if 'all' in basin:
        da_mask_atlantic_regrid = da_mask_atlantic_regrid.where(da_mask_atlantic_regrid>0.)
        da_mask_atlantic_regrid = da_mask_atlantic_regrid.where(da_mask_atlantic_regrid.isnull(),other=1.)

        da_mask_indian_regrid = da_mask_indian_regrid.where(da_mask_indian_regrid>0.)
        da_mask_indian_regrid = da_mask_indian_regrid.where(da_mask_indian_regrid.isnull(),other=1.)

        da_mask_pacific_regrid = da_mask_pacific_regrid.where(da_mask_pacific_regrid>0.)
        da_mask_pacific_regrid = da_mask_pacific_regrid.where(da_mask_pacific_regrid.isnull(),other=1.)
        
        da_3basin = da_3basin.where(da_3basin>0.)
        da_3basin = da_3basin.where(da_3basin.isnull(),other=1.)

        da_indopac = da_indopac.where(da_indopac>0.)
        da_indopac = da_indopac.where(da_indopac.isnull(),other=1.)
        
        return da_mask_atlantic_regrid,da_mask_indian_regrid,da_mask_pacific_regrid,da_3basin,da_indopac
    
    


def mom6_bathymetry(da_var=None):
    """
    The function read the deptho variable in MOM6 ocean model output to 
    create the bathymetry mask

    Parameters:
    
        da_var(xr.DataArray) - the 2D var one want the da_bathy to regrid to. Default 
        to output on original tracer grid in MOM6. 
        
    Returns:
        
        da_bathy (xr.DataArray) 
        
    """
    
    
    # topo in MOM6
    bathy_dir = '/storage2/chiaweih/OMIP/GFDL/JRA/OM4p25_JRA55do1.4_0netfw_cycle6/'
    bathy_file = 'ocean_monthly.static.nc'
    da_bathy = xr.open_dataset(bathy_dir+bathy_file)
    
    
    
    ds_bathy = xr.Dataset(coords={'lon':(('yh','xh'),da_bathy.geolon.values),
                                  'lat':(('yh','xh'),da_bathy.geolat.values),
                                  'yh' :da_bathy.yh.values,
                                  'xh' :da_bathy.xh.values})
    ds_bathy['deptho'] = da_bathy.deptho
        
    ds_bathy = ds_bathy.rename({'xh':'x','yh':'y'})
    
    
    if da_var is None:
        return ds_bathy
    else:
        # Regridding to the tracer points
        regridder = xe.Regridder(ds_bathy,
                                 da_var,
                                 'bilinear',
                                 filename='bathy_regrid.nc',
                                 periodic=True)
        da_bathy_regrid = regridder(ds_bathy.deptho)
        da_bathy_regrid['x'] = da_var.x
        da_bathy_regrid['y'] = da_var.y
        regridder.clean_weight_file()
        
        return da_bathy_regrid
        
        


def mom6_bathymetry_basin_mzonal(da_mask=None):
    """
    The function read the deptho variable in MOM6 ocean model output to 
    create the basin zonal mean bathymetry mask

    Parameters:
    
        da_mask(xr.DataArray) - the mask for the ocean basin one want 
        to calculate the zonal mean.
        
    Returns:
        
        da_bathy_mzonal (xr.DataArray) 
        
    """

    da_bathy = mom6_bathymetry(da_var=da_mask)
    
    if da_mask is None:
        da_bathy_mzonal =  da_bathy.mean(dim='x')
    else:
        da_bathy_mzonal = (da_bathy*da_mask).mean(dim='x')
    
    return da_bathy_mzonal
