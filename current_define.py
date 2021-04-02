import xarray as xr
import numpy as np

# constant 
sea_density = 1025.              # kg/m^3
sea_heatcap = 3991.86795711963   # J/(kg K)
r_earth = 6.371E8                # cm

def delta_dist(ds_transect,transect_plane,xyz_name):
    """
    Calculate the dy, dx, dz, for different transect
    
    """
    
    if transect_plane in ['xy']:
        # calculate dx 
        da_dx = ds_transect[xyz_name[0]].copy()
        da_dx.values[1:-1] = (ds_transect[xyz_name[0]][:-1].diff(xyz_name[0],1).values+\
                              ds_transect[xyz_name[0]][1:].diff(xyz_name[0],1).values)/2.
        da_dx.values[0] = (ds_transect[xyz_name[0]][1]-ds_transect[xyz_name[0]][0]).values
        da_dx.values[-1] = (ds_transect[xyz_name[0]][-1]-ds_transect[xyz_name[0]][-2]).values
        da_dx = da_dx/180.*np.pi*r_earth*np.cos(ds_transect[xyz_name[1]]/180.*np.pi)/100.     # meters
        
        # calculate dy 
        da_dy = ds_transect[xyz_name[1]].copy()
        da_dy.values[1:-1] = (ds_transect[xyz_name[1]][:-1].diff(xyz_name[1],1).values+\
                              ds_transect[xyz_name[1]][1:].diff(xyz_name[1],1).values)/2.
        da_dy.values[0] = (ds_transect[xyz_name[1]][1]-ds_transect[xyz_name[1]][0]).values
        da_dy.values[-1] = (ds_transect[xyz_name[1]][-1]-ds_transect[xyz_name[1]][-2]).values
        da_dy = da_dy/180.*np.pi*r_earth/100.                                                  # meters
        
        return da_dx,da_dy
        
    elif transect_plane in ['xz']:
        
        # calculate dx 
        da_dx = ds_transect[xyz_name[0]].copy()
        da_dx.values[1:-1] = (ds_transect[xyz_name[0]][:-1].diff(xyz_name[0],1).values+\
                              ds_transect[xyz_name[0]][1:].diff(xyz_name[0],1).values)/2.
        da_dx.values[0] = (ds_transect[xyz_name[0]][1]-ds_transect[xyz_name[0]][0]).values
        da_dx.values[-1] = (ds_transect[xyz_name[0]][-1]-ds_transect[xyz_name[0]][-2]).values
        da_dx = da_dx/180.*np.pi*r_earth*np.cos(ds_transect[xyz_name[1]]/180.*np.pi)/100.                # meters
                                                                      
        # calculate dz
        da_dz = ds_transect[xyz_name[2]].copy()
        da_dz.values[1:-1] = (ds_transect[xyz_name[2]][:-1].diff(xyz_name[2],1).values+\
                              ds_transect[xyz_name[2]][1:].diff(xyz_name[2],1).values)/2.
        da_dz.values[0] = (ds_transect[xyz_name[2]][1]-ds_transect[xyz_name[2]][0]).values
        da_dz.values[-1] = (ds_transect[xyz_name[2]][-1]-ds_transect[xyz_name[2]][-2]).values  # meters 
        
        return da_dx,da_dz
                                                                      
    elif transect_plane in ['yz']:
        
        # calculate dy 
        da_dy = ds_transect[xyz_name[1]].copy()
        da_dy.values[1:-1] = (ds_transect[xyz_name[1]][:-1].diff(xyz_name[1],1).values+\
                              ds_transect[xyz_name[1]][1:].diff(xyz_name[1],1).values)/2.
        da_dy.values[0] = (ds_transect[xyz_name[1]][1]-ds_transect[xyz_name[1]][0]).values
        da_dy.values[-1] = (ds_transect[xyz_name[1]][-1]-ds_transect[xyz_name[1]][-2]).values
        da_dy = da_dy/180.*np.pi*r_earth/100.                                                  # meters
        
        # calculate dz
        da_dz = ds_transect[xyz_name[2]].copy()
        da_dz.values[1:-1] = (ds_transect[xyz_name[2]][:-1].diff(xyz_name[2],1).values+\
                              ds_transect[xyz_name[2]][1:].diff(xyz_name[2],1).values)/2.
        da_dz.values[0] = (ds_transect[xyz_name[2]][1]-ds_transect[xyz_name[2]][0]).values
        da_dz.values[-1] = (ds_transect[xyz_name[2]][-1]-ds_transect[xyz_name[2]][-2]).values  # meters
        
        return da_dy, da_dz
       



###########################################################
def transect_yz_180(ds_transect,transect_plane,xyz_name,ebox_tmean,wbox_tmean) :       

    current = xr.Dataset()

    # define region of current 
    euc_lat = [-2.5, 2.5]
    euc_dep = [0, 400]

    necc_lat = [2.5,10]
    necc_dep = [0, 400]

    secc_lat = [-10, -2.5]
    secc_dep = [0, 400]

    nec_lat = [6, 20]
    nec_dep = [0, 400]

    sec_lat  = [-20, 6]
    sec_dep  = [0, 400]
    
    nsec_lat  = [0, 6]
    nsec_dep  = [0, 400]

    res1_lat = [-20, -10]
    res1_dep = [0, 400]

    res2_lat = [6, 20]
    res2_dep = [0, 400]

    current_list = ['euc','necc','secc','nec','sec','res1','res2','nsec']
    cur_meandirect_list = ['e','e','e','w','w','e','e','w']
    cur_range_list = [[euc_lat,euc_dep],
                      [necc_lat,necc_dep],
                      [secc_lat,secc_dep],
                      [nec_lat,nec_dep],
                      [sec_lat,sec_dep],
                      [res1_lat,res1_dep],
                      [res2_lat,res2_dep],
                      [nsec_lat,nsec_dep]]


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_uo = ds_transect['uo_mean']

    for ncurr,curr in enumerate(current_list):
        if cur_meandirect_list[ncurr] in ['e']:
            dict_curr_mask[curr] = mean_uo.where(\
                           (mean_uo[xyz_name[1]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[1]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_uo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_uo > 0.)\
                           ,drop=True)*0.+1
        elif cur_meandirect_list[ncurr] in ['w']:
            dict_curr_mask[curr] = mean_uo.where(\
                           (mean_uo[xyz_name[1]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[1]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_uo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_uo < 0.)\
                           ,drop=True)*0.+1    


    da_htran_e = (ds_transect['thetao']-ebox_tmean)*ds_transect['uo']
    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['uo']

    da_htran_Tv_e = (ds_transect['thetao_mean']-ebox_tmean.mean())*\
                    (ds_transect['uo']-ds_transect['uo_mean'])
    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['uo']-ds_transect['uo_mean'])

    da_htran_Vt_e = ds_transect['uo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (ebox_tmean-ebox_tmean.mean()))
    da_htran_Vt_w = ds_transect['uo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dy,da_dz = delta_dist(ds_transect,transect_plane,xyz_name)

    for ncurr,curr in enumerate(current_list):
        current['ebox_'+curr] = (da_htran_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Tv_'+curr] = (da_htran_Tv_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Vt_'+curr] = (da_htran_Vt_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['uo']*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                               *1.E-6  # Sv
        
    return current

###########################################################
def transect_yz_160w140w(ds_transect,transect_plane,xyz_name,ebox_tmean,wbox_tmean) :       

    current = xr.Dataset()

    # define region of current 
    euc_lat = [-2.5, 2.5]
    euc_dep = [0, 400]

    necc_lat = [2.5,10]
    necc_dep = [0, 400]

    secc_lat = [-10, -2.5]
    secc_dep = [0, 400]

    nec_lat = [6, 20]
    nec_dep = [0, 400]

    sec_lat  = [-20, 6]
    sec_dep  = [0, 400]
    
    nsec_lat  = [0, 6]
    nsec_dep  = [0, 400]

    res1_lat = [-20, -10]
    res1_dep = [0, 400]

    res2_lat = [6, 20]
    res2_dep = [0, 400]

    current_list = ['euc','necc','secc','nec','sec','res1','res2','nsec']
    cur_meandirect_list = ['e','e','e','w','w','e','e','w']
    cur_range_list = [[euc_lat,euc_dep],
                      [necc_lat,necc_dep],
                      [secc_lat,secc_dep],
                      [nec_lat,nec_dep],
                      [sec_lat,sec_dep],
                      [res1_lat,res1_dep],
                      [res2_lat,res2_dep],
                      [nsec_lat,nsec_dep]]


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_uo = ds_transect['uo_mean']

    for ncurr,curr in enumerate(current_list):
        if cur_meandirect_list[ncurr] in ['e']:
            dict_curr_mask[curr] = mean_uo.where(\
                           (mean_uo[xyz_name[1]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[1]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_uo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_uo > 0.)\
                           ,drop=True)*0.+1
        elif cur_meandirect_list[ncurr] in ['w']:
            dict_curr_mask[curr] = mean_uo.where(\
                           (mean_uo[xyz_name[1]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[1]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_uo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_uo < 0.)\
                           ,drop=True)*0.+1    


    da_htran_e = (ds_transect['thetao']-ebox_tmean)*ds_transect['uo']
    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['uo']

    da_htran_Tv_e = (ds_transect['thetao_mean']-ebox_tmean.mean())*\
                    (ds_transect['uo']-ds_transect['uo_mean'])
    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['uo']-ds_transect['uo_mean'])

    da_htran_Vt_e = ds_transect['uo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (ebox_tmean-ebox_tmean.mean()))
    da_htran_Vt_w = ds_transect['uo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dy,da_dz = delta_dist(ds_transect,transect_plane,xyz_name)

    for ncurr,curr in enumerate(current_list):
        current['ebox_'+curr] = (da_htran_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Tv_'+curr] = (da_htran_Tv_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Vt_'+curr] = (da_htran_Vt_e*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['uo']*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                               *1.E-6  # Sv
        
    return current


        
        
###########################################################
def transect_yz_120e(ds_transect,transect_plane,xyz_name,wbox_tmean) :       

    current = xr.Dataset()
    # define region of current 
    reg1_lat = [15,20]
    reg1_dep = [0, 400]

    reg2_lat = [1,15]
    reg2_dep = [0, 400]

    reg3_lat = [-5, 1]
    reg3_dep = [0, 400]

    reg4_lat = [-10, -5]
    reg4_dep = [0, 400]

    reg5_lat  = [-20, -10]
    reg5_dep  = [0, 400]

    current_list = ['reg1','reg2','reg3','reg4','reg5']
    cur_range_list = [[reg1_lat,reg1_dep],
                      [reg2_lat,reg2_dep],
                      [reg3_lat,reg3_dep],
                      [reg4_lat,reg4_dep],
                      [reg5_lat,reg5_dep]]


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_uo = ds_transect['uo_mean']

    for ncurr,curr in enumerate(current_list):
        dict_curr_mask[curr] = mean_uo.where(\
                           (mean_uo[xyz_name[1]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[1]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_uo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_uo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))\
                           ,drop=True)*0.+1   

    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['uo']

    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['uo']-ds_transect['uo_mean'])

    da_htran_Vt_w = ds_transect['uo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dy,da_dz = delta_dist(ds_transect,transect_plane,xyz_name)
    
    for ncurr,curr in enumerate(current_list):
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt

        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['uo']*dict_curr_mask[curr]*da_dy*da_dz).sum(dim=[xyz_name[1],xyz_name[2]])\
                               *1.E-6  # Sv        
        
    return current


###########################################################
def transect_xz_20n(ds_transect,transect_plane,xyz_name,ebox_tmean,wbox_tmean) :       

    current = xr.Dataset()

    # define region of current 
    reg1_lon = [120,180]     # >0
    reg1_dep = [0, 400]

    reg11_lon  = [120, 123]   # >0
    reg11_dep  = [0, 400]

    reg2_lon = [120,180]     # <0
    reg2_dep = [0, 400]

    reg3_lon = [180, -100]   # >0
    reg3_dep = [0, 400]

    reg4_lon = [180, -100]   # <0
    reg4_dep = [0, 400]

    reg5_lon  = [100, 120]   # out of box
    reg5_dep  = [0, 400]
    
    current_list = ['reg1','reg11','reg2','reg3','reg4','reg5']
    cur_meandirect_list = ['n','n','s','n','s','']
    cur_range_list = [[reg1_lon,reg1_dep],
                      [reg11_lon,reg11_dep],
                      [reg2_lon,reg2_dep],
                      [reg3_lon,reg3_dep],
                      [reg4_lon,reg4_dep],
                      [reg5_lon,reg5_dep]]
    
    
    # correct the lon range
    for ncurr,curr in enumerate(current_list):
        lon_mod = np.array(cur_range_list[ncurr][0])
        lonmin = ds_transect.lon.min()
        ind1 = np.where(lon_mod>np.float(360.+lonmin))[0]
        lon_mod[ind1] = lon_mod[ind1]-360.
        cur_range_list[ncurr][0] = lon_mod
        # change Lon range to -300-60 (might be different for different model) 


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_vo = ds_transect['vo_mean']

    for ncurr,curr in enumerate(current_list):
        if cur_meandirect_list[ncurr] in ['n']:
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_vo > 0.)\
                           ,drop=True)*0.+1
        elif cur_meandirect_list[ncurr] in ['s']:
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_vo < 0.)\
                           ,drop=True)*0.+1    
        else :
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))\
                           ,drop=True)*0.+1  


    da_htran_e = (ds_transect['thetao']-ebox_tmean)*ds_transect['vo']
    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['vo']

    da_htran_Tv_e = (ds_transect['thetao_mean']-ebox_tmean.mean())*\
                    (ds_transect['vo']-ds_transect['vo_mean'])
    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['vo']-ds_transect['vo_mean'])

    da_htran_Vt_e = ds_transect['vo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (ebox_tmean-ebox_tmean.mean()))
    da_htran_Vt_w = ds_transect['vo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dx,da_dz = delta_dist(ds_transect,transect_plane,xyz_name)

    for ncurr,curr in enumerate(current_list):
        current['ebox_'+curr] = (da_htran_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Tv_'+curr] = (da_htran_Tv_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Vt_'+curr] = (da_htran_Vt_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['vo']*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                               *1.E-6  # Sv
        
    return current


###########################################################
def transect_xz_20s(ds_transect,transect_plane,xyz_name,ebox_tmean,wbox_tmean) :       

    current = xr.Dataset()

    # define region of current 
    reg1_lon = [120,180]     # >0
    reg1_dep = [0, 400]

    reg11_lon  = [150, 160]   # <0
    reg11_dep  = [0, 400]

    reg2_lon = [120,180]     # <0
    reg2_dep = [0, 400]

    reg3_lon = [180, -60]   # >0
    reg3_dep = [0, 400]

    reg4_lon = [180, -60]   # <0
    reg4_dep = [0, 400]

    reg5_lon  = [100, 120]   # out of box
    reg5_dep  = [0, 400]
    
    current_list = ['reg1','reg11','reg2','reg3','reg4','reg5']
    cur_meandirect_list = ['n','s','s','n','s','']
    cur_range_list = [[reg1_lon,reg1_dep],
                      [reg11_lon,reg11_dep],
                      [reg2_lon,reg2_dep],
                      [reg3_lon,reg3_dep],
                      [reg4_lon,reg4_dep],
                      [reg5_lon,reg5_dep]]
    
    
    # correct the lon range
    for ncurr,curr in enumerate(current_list):
        lon_mod = np.array(cur_range_list[ncurr][0])
        lonmin = ds_transect.lon.min()
        ind1 = np.where(lon_mod>np.float(360.+lonmin))[0]
        lon_mod[ind1] = lon_mod[ind1]-360.
        cur_range_list[ncurr][0] = lon_mod
        # change Lon range to -300-60 (might be different for different model) 


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_vo = ds_transect['vo_mean']

    for ncurr,curr in enumerate(current_list):
        if cur_meandirect_list[ncurr] in ['n']:
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_vo > 0.)\
                           ,drop=True)*0.+1
        elif cur_meandirect_list[ncurr] in ['s']:
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_vo < 0.)\
                           ,drop=True)*0.+1    
        else :
            dict_curr_mask[curr] = mean_vo.where(\
                           (mean_vo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_vo[xyz_name[2]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_vo[xyz_name[2]] < np.max(cur_range_list[ncurr][1]))\
                           ,drop=True)*0.+1  


    da_htran_e = (ds_transect['thetao']-ebox_tmean)*ds_transect['vo']
    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['vo']

    da_htran_Tv_e = (ds_transect['thetao_mean']-ebox_tmean.mean())*\
                    (ds_transect['vo']-ds_transect['vo_mean'])
    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['vo']-ds_transect['vo_mean'])

    da_htran_Vt_e = ds_transect['vo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (ebox_tmean-ebox_tmean.mean()))
    da_htran_Vt_w = ds_transect['vo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dx,da_dz = delta_dist(ds_transect,transect_plane,xyz_name)

    for ncurr,curr in enumerate(current_list):
        current['ebox_'+curr] = (da_htran_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Tv_'+curr] = (da_htran_Tv_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Vt_'+curr] = (da_htran_Vt_e*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['vo']*dict_curr_mask[curr]*da_dx*da_dz).sum(dim=[xyz_name[0],xyz_name[2]])\
                               *1.E-6  # Sv
        
    return current

###########################################################
def transect_xy_400m(ds_transect,transect_plane,xyz_name,ebox_tmean,wbox_tmean) :       

    current = xr.Dataset()

    # define region of current 
    reg1_lon  = [120,180]     # >0
    reg1_lat  = [-20, 20]

    reg11_lon = [120, 180]    # <0
    reg11_lat = [-20, 20]

    reg2_lon  = [180,-60]     # >0
    reg2_lat  = [-20, 20]

    reg21_lon = [180,-60]     # <0
    reg21_lat = [-20, 20]
    
    current_list = ['reg1','reg11','reg2','reg21']
    cur_meandirect_list = ['u','d','u','d','s','']
    cur_range_list = [[reg1_lon,reg1_lat],
                      [reg11_lon,reg11_lat],
                      [reg2_lon,reg2_lat],
                      [reg21_lon,reg21_lat]]
  
    
    # correct the lon range
    for ncurr,curr in enumerate(current_list):
        lon_mod = np.array(cur_range_list[ncurr][0])
        lonmin = ds_transect.lon.min()
        ind1 = np.where(lon_mod>np.float(360.+lonmin))[0]
        lon_mod[ind1] = lon_mod[ind1]-360.
        cur_range_list[ncurr][0] = lon_mod
        # change Lon range to -300-60 (might be different for different model) 


    # determine the current mask based on mean field
    dict_curr_mask = {}
    mean_wo = ds_transect['wo_mean']

    for ncurr,curr in enumerate(current_list):
        if cur_meandirect_list[ncurr] in ['u']:
            dict_curr_mask[curr] = mean_wo.where(\
                           (mean_wo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_wo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_wo[xyz_name[1]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_wo[xyz_name[1]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_wo > 0.)\
                           ,drop=True)*0.+1
        elif cur_meandirect_list[ncurr] in ['d']:
            dict_curr_mask[curr] = mean_wo.where(\
                           (mean_wo[xyz_name[0]] > np.min(cur_range_list[ncurr][0]))&\
                           (mean_wo[xyz_name[0]] < np.max(cur_range_list[ncurr][0]))&\
                           (mean_wo[xyz_name[1]] > np.min(cur_range_list[ncurr][1]))&\
                           (mean_wo[xyz_name[1]] < np.max(cur_range_list[ncurr][1]))&\
                           (mean_wo < 0.)\
                           ,drop=True)*0.+1    


    da_htran_e = (ds_transect['thetao']-ebox_tmean)*ds_transect['wo']
    da_htran_w = (ds_transect['thetao']-wbox_tmean)*ds_transect['wo']

    da_htran_Tv_e = (ds_transect['thetao_mean']-ebox_tmean.mean())*\
                    (ds_transect['wo']-ds_transect['wo_mean'])
    da_htran_Tv_w = (ds_transect['thetao_mean']-wbox_tmean.mean())*\
                    (ds_transect['wo']-ds_transect['wo_mean'])

    da_htran_Vt_e = ds_transect['wo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (ebox_tmean-ebox_tmean.mean()))
    da_htran_Vt_w = ds_transect['wo_mean']*\
                            ((ds_transect['thetao']-ds_transect['thetao_mean'])-\
                             (wbox_tmean-wbox_tmean.mean()))

    da_dx,da_dy = delta_dist(ds_transect,transect_plane,xyz_name)

    for ncurr,curr in enumerate(current_list):
        current['ebox_'+curr] = (da_htran_e*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_'+curr] = (da_htran_w*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Tv_'+curr] = (da_htran_Tv_e*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Tv_'+curr] = (da_htran_Tv_w*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['ebox_Vt_'+curr] = (da_htran_Vt_e*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt
        current['wbox_Vt_'+curr] = (da_htran_Vt_w*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                                 *sea_density*sea_heatcap*1.E-15  # Petawatt 
        current[curr+'_Sv']  = (ds_transect['wo']*dict_curr_mask[curr]*da_dx*da_dy).sum(dim=[xyz_name[0],xyz_name[1]])\
                               *1.E-6  # Sv
        
    return current
