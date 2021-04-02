#!/usr/bin/env python
"""
The script generate the heat budget analysis bar chart
and current heat advection bar chart

in the form of
figure 6abc : mean state
figure 19 : seasonal variability
in the paper.

input files
============
JRA55-do/CORE : regional averaged (net_heat_coupler)
                (exec io_gfdl_regionsum.py first)
                volume integral   (heat content)
                (exec io_gfdl_Tmean_Hcont.py first)
                transect    (uo,vo,thetao,wo)
                (exec io_gfdl_w.py first)
                (exec io_gfdl_transect.py second)


function used
==================
create_ocean_mask.levitus98 : which generate the Pacific basin mask
spherical_area.cal_area     : generate area array based on the lon lat of data


"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


# # Read regional mean
reg_mean_ts_mlist = {}
reg_mean_season_ts_mlist = {}
model_list = []

syear = 1958
fyear = 2007

#####################
model = 'JRA'
model_list.append(model)

# JRA
varnames = ['srfflux','hcont']

paths = ['./data/GFDL/JRA/regional_avg/',
         './data/GFDL/JRA/regional_avg/']

files = ['JRA_net_heat_coupler_regional_sum_ts_scpt.nc',
        'JRA_heatcont_scpt_400.nc']


reg_mean_ts_list={}
reg_mean_season_ts_list={}
for nvar,varname in enumerate(varnames):
    reg_mean_ts_list[varname] = xr.open_dataset(paths[nvar]+files[nvar])
    reg_mean_ts_list[varname] = reg_mean_ts_list[varname]\
                                .where((reg_mean_ts_list[varname]['time.year']>=syear)&\
                                       (reg_mean_ts_list[varname]['time.year']<=fyear),\
                                        drop=True)
reg_mean_ts_mlist[model] = reg_mean_ts_list
reg_mean_season_ts_mlist[model] = reg_mean_season_ts_list

#####################
model = 'CORE'
model_list.append(model)

# CORE
varnames = ['srfflux','hcont']

paths = ['./data/GFDL/CORE/regional_avg/',
         './data/GFDL/CORE/regional_avg/']

files = ['CORE_net_heat_coupler_regional_sum_ts_scpt.nc',
        'CORE_heatcont_scpt_400.nc']

reg_mean_ts_list={}
reg_mean_season_ts_list={}
for nvar,varname in enumerate(varnames):
    reg_mean_ts_list[varname] = xr.open_dataset(paths[nvar]+files[nvar])
    reg_mean_ts_list[varname] = reg_mean_ts_list[varname]\
                                .where((reg_mean_ts_list[varname]['time.year']>=syear)&\
                                       (reg_mean_ts_list[varname]['time.year']<=fyear),\
                                        drop=True)
reg_mean_ts_mlist[model] = reg_mean_ts_list
reg_mean_season_ts_mlist[model] = reg_mean_season_ts_list


# # Regional mean processing
for model in model_list:
    for nvar,varname in enumerate(varnames):
        # remove seasonal
        da_mean = reg_mean_ts_mlist[model][varname].mean()
        da_seasonal = (reg_mean_ts_mlist[model][varname]-da_mean).groupby('time.month').mean(dim='time')
        reg_mean_season_ts_mlist[model][varname]=da_seasonal
        reg_mean_ts_mlist[model][varname] = \
                reg_mean_ts_mlist[model][varname].groupby('time.month')-da_seasonal
        # 13 month smoothing
        reg_mean_ts_mlist[model][varname] = reg_mean_ts_mlist[model][varname]\
                                             .rolling(time=13, center=True).mean(dim='time')



# # Read transect current
current_mlist = {}
current_season_mlist = {}
model_list = []

#####################
model = 'JRA'
model_list.append(model)

# JRA
trannames = ['180','120','20','-20','400','-160']

paths = ['./data/GFDL/JRA/transect_scpt/JRA_uo_thetao_yz_-180/',
         './data/GFDL/JRA/transect_scpt/JRA_uo_thetao_yz_120/',
         './data/GFDL/JRA/transect_scpt/JRA_vo_thetao_xz_20/',
         './data/GFDL/JRA/transect_scpt/JRA_vo_thetao_xz_-20/',
         './data/GFDL/JRA/transect_scpt/JRA_wo_thetao_xy_400/',
         './data/GFDL/JRA/transect_scpt/JRA_uo_thetao_yz_-160/']

files = ['current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc']


current_list={}
current_season_list={}
for ntran,tranname in enumerate(trannames):
    current_list[tranname] = xr.open_dataset(paths[ntran]+files[ntran])
    current_list[tranname] = current_list[tranname]                                .where((current_list[tranname]['time.year']>=syear)&
                                       (current_list[tranname]['time.year']<=fyear),
                                        drop=True)
current_mlist[model] = current_list
current_season_mlist[model] = current_season_list

#####################
model = 'CORE'
model_list.append(model)

# CORE
trannames = ['180','120','20','-20','400','-160']

paths = ['./data/GFDL/CORE/transect_scpt/CORE_uo_thetao_yz_-180/',
         './data/GFDL/CORE/transect_scpt/CORE_uo_thetao_yz_120/',
         './data/GFDL/CORE/transect_scpt/CORE_vo_thetao_xz_20/',
         './data/GFDL/CORE/transect_scpt/CORE_vo_thetao_xz_-20/',
         './data/GFDL/CORE/transect_scpt/CORE_wo_thetao_xy_400/',
         './data/GFDL/CORE/transect_scpt/CORE_uo_thetao_yz_-160/',]

files = ['current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc',
         'current_ts.nc']


current_list={}
current_season_list={}
for ntran,tranname in enumerate(trannames):
    current_list[tranname] = xr.open_dataset(paths[ntran]+files[ntran])
    current_list[tranname] = current_list[tranname]\
                                .where((current_list[tranname]['time.year']>=syear)&\
                                       (current_list[tranname]['time.year']<=fyear),
                                        drop=True)
current_mlist[model] = current_list
current_season_mlist[model] = current_season_list


# # Current processing
for model in model_list:
    for ntran,tranname in enumerate(trannames):
        # remove seasonal
        da_mean = current_mlist[model][tranname].mean()
        da_seasonal = (current_mlist[model][tranname]-da_mean).groupby('time.month').mean(dim='time')
        current_season_mlist[model][tranname] = da_seasonal
        current_mlist[model][tranname] =  \
        current_mlist[model][tranname].groupby('time.month')-da_seasonal
        # 13 month smoothing
        current_mlist[model][tranname] = current_mlist[model][tranname].rolling(time=13, center=True).mean(dim='time')




########################################################################################################
# # Mean bar chart
from time_series_analyses import ts_mean_conf
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10,20))
################################################################
fig.text(-0.1,0.15,'a',size=30)
ax = fig.add_axes([0,0,1.,0.15])

model = 'JRA'
box = 'wbox'

var2w = -(current_mlist[model]['180']['%s_euc'%(box)]+
         current_mlist[model]['180']['%s_necc'%(box)]+
         current_mlist[model]['180']['%s_secc'%(box)]+
         current_mlist[model]['180']['%s_nec'%(box)]+
         current_mlist[model]['180']['%s_sec'%(box)]+
         current_mlist[model]['180']['%s_res1'%(box)]+
         current_mlist[model]['180']['%s_res2'%(box)])

var7w = +(current_mlist[model]['120']['%s_reg1'%(box)]+
         current_mlist[model]['120']['%s_reg2'%(box)]+
         current_mlist[model]['120']['%s_reg3'%(box)]+
         current_mlist[model]['120']['%s_reg4'%(box)]+
         current_mlist[model]['120']['%s_reg5'%(box)])

var3w = -(current_mlist[model]['20']['%s_reg1'%(box)]+
         current_mlist[model]['20']['%s_reg2'%(box)])

var4w = +(current_mlist[model]['-20']['%s_reg1'%(box)]+
         current_mlist[model]['-20']['%s_reg2'%(box)])

var5w = +(current_mlist[model]['400']['%s_reg1'%(box)]+
         current_mlist[model]['400']['%s_reg11'%(box)])  # PW

var6w = (reg_mean_ts_mlist[model]['srfflux']['net_heat_coupler_120_180_-20_20'])*1e-15 # W => PW

var1w = var2w+var3w+var4w+var5w+var6w+var7w

totw = reg_mean_ts_mlist[model]['hcont']['heat_content_120_180']\
            .differentiate('time',datetime_unit='s')*1e-15

var8w = totw-var1w


box = 'ebox'

var2e = +(current_mlist[model]['180']['%s_euc'%(box)]+
         current_mlist[model]['180']['%s_necc'%(box)]+
         current_mlist[model]['180']['%s_secc'%(box)]+
         current_mlist[model]['180']['%s_nec'%(box)]+
         current_mlist[model]['180']['%s_sec'%(box)]+
         current_mlist[model]['180']['%s_res1'%(box)]+
         current_mlist[model]['180']['%s_res2'%(box)])

var3e = -(current_mlist[model]['20']['%s_reg3'%(box)]+
         current_mlist[model]['20']['%s_reg4'%(box)])

var4e = +(current_mlist[model]['-20']['%s_reg3'%(box)]+
         current_mlist[model]['-20']['%s_reg4'%(box)])

var5e = +(current_mlist[model]['400']['%s_reg2'%(box)]+
         current_mlist[model]['400']['%s_reg21'%(box)])  # PW

var6e = (reg_mean_ts_mlist[model]['srfflux']['net_heat_coupler_180_-60_-20_20'])*1e-15 # W => PW

var1e = var2e+var3e+var4e+var5e+var6e

tote = reg_mean_ts_mlist[model]['hcont']['heat_content_180_-60']\
            .differentiate('time',datetime_unit='s')*1e-15

var8e = tote-var1e



var2w_conf=ts_mean_conf(var2w)
var3w_conf=ts_mean_conf(var3w)
var4w_conf=ts_mean_conf(var4w)
var5w_conf=ts_mean_conf(var5w)
var6w_conf=ts_mean_conf(var6w)
var7w_conf=ts_mean_conf(var7w)
var1w_conf=np.sqrt(var2w_conf**2+var3w_conf**2+var4w_conf**2+var5w_conf**2+var6w_conf**2+var7w_conf**2)
totw_conf=ts_mean_conf(totw)
var8w_conf=np.sqrt(var1w_conf**2+totw_conf**2)


var2e_conf=ts_mean_conf(var2e)
var3e_conf=ts_mean_conf(var3e)
var4e_conf=ts_mean_conf(var4e)
var5e_conf=ts_mean_conf(var5e)
var6e_conf=ts_mean_conf(var6e)
var1e_conf=np.sqrt(var2e_conf**2+var3e_conf**2+var4e_conf**2+var5e_conf**2+var6e_conf**2)
tote_conf=ts_mean_conf(tote)
var8e_conf=np.sqrt(var1e_conf**2+tote_conf**2)


labels = ['120$^\circ$E','180$^\circ$','20$^\circ$N', '20$^\circ$S','400m','SRF','RES','TOT']
ebox_means = [0.,           var2e.mean(), var3e.mean(), var4e.mean(), var5e.mean(), var6e.mean(), var8e.mean(),tote.mean()]
wbox_means = [var7w.mean(), var2w.mean(), var3w.mean(), var4w.mean(), var5w.mean(), var6w.mean(), var8w.mean(),totw.mean()]
ebox_conf  = [0.,         var2e_conf, var3e_conf, var4e_conf, var5e_conf, var6e_conf, var8e_conf,0]
wbox_conf  = [var7w_conf, var2w_conf, var3w_conf, var4w_conf, var5w_conf, var6w_conf, var8w_conf,0]


x = np.arange(len(labels))  # the label locations
width = 0.2                # the width of the bars

rects2 = ax.bar(x - width*(1.+1./2.), wbox_means, width, yerr=wbox_conf, label='Wbox %s55-do'%model, color='C1')
rects1 = ax.bar(x + width*(1./2.), ebox_means, width, yerr=ebox_conf, label='Ebox %s55-do'%model, color='C0')

###########################################
model = 'CORE'
box = 'wbox'

var2w = -(current_mlist[model]['180']['%s_euc'%(box)]+
         current_mlist[model]['180']['%s_necc'%(box)]+
         current_mlist[model]['180']['%s_secc'%(box)]+
         current_mlist[model]['180']['%s_nec'%(box)]+
         current_mlist[model]['180']['%s_sec'%(box)]+
         current_mlist[model]['180']['%s_res1'%(box)]+
         current_mlist[model]['180']['%s_res2'%(box)])

var7w = +(current_mlist[model]['120']['%s_reg1'%(box)]+
         current_mlist[model]['120']['%s_reg2'%(box)]+
         current_mlist[model]['120']['%s_reg3'%(box)]+
         current_mlist[model]['120']['%s_reg4'%(box)]+
         current_mlist[model]['120']['%s_reg5'%(box)])

var3w = -(current_mlist[model]['20']['%s_reg1'%(box)]+
         current_mlist[model]['20']['%s_reg2'%(box)])

var4w = +(current_mlist[model]['-20']['%s_reg1'%(box)]+
         current_mlist[model]['-20']['%s_reg2'%(box)])

var5w = +(current_mlist[model]['400']['%s_reg1'%(box)]+
         current_mlist[model]['400']['%s_reg11'%(box)])  # PW

var6w = (reg_mean_ts_mlist[model]['srfflux']['net_heat_coupler_120_180_-20_20'])*1e-15 # W => PW

var1w = var2w+var3w+var4w+var5w+var6w+var7w

totw= reg_mean_ts_mlist[model]['hcont']['heat_content_120_180']\
            .differentiate('time',datetime_unit='s')*1e-15

var8w = totw-var1w


box = 'ebox'

var2e = +(current_mlist[model]['180']['%s_euc'%(box)]+
         current_mlist[model]['180']['%s_necc'%(box)]+
         current_mlist[model]['180']['%s_secc'%(box)]+
         current_mlist[model]['180']['%s_nec'%(box)]+
         current_mlist[model]['180']['%s_sec'%(box)]+
         current_mlist[model]['180']['%s_res1'%(box)]+
         current_mlist[model]['180']['%s_res2'%(box)])

var3e = -(current_mlist[model]['20']['%s_reg3'%(box)]+
         current_mlist[model]['20']['%s_reg4'%(box)])

var4e = +(current_mlist[model]['-20']['%s_reg3'%(box)]+
         current_mlist[model]['-20']['%s_reg4'%(box)])

var5e = +(current_mlist[model]['400']['%s_reg2'%(box)]+
         current_mlist[model]['400']['%s_reg21'%(box)])  # PW

var6e = (reg_mean_ts_mlist[model]['srfflux']['net_heat_coupler_180_-60_-20_20'])*1e-15 # W => PW

var1e = var2e+var3e+var4e+var5e+var6e

tote = reg_mean_ts_mlist[model]['hcont']['heat_content_180_-60']\
            .differentiate('time',datetime_unit='s')*1e-15

var8e = tote-var1e



var2w_conf=ts_mean_conf(var2w)
var3w_conf=ts_mean_conf(var3w)
var4w_conf=ts_mean_conf(var4w)
var5w_conf=ts_mean_conf(var5w)
var6w_conf=ts_mean_conf(var6w)
var7w_conf=ts_mean_conf(var7w)
var1w_conf=np.sqrt(var2w_conf**2+var3w_conf**2+var4w_conf**2+var5w_conf**2+var6w_conf**2+var7w_conf**2)
totw_conf=ts_mean_conf(totw)
var8w_conf=np.sqrt(var1w_conf**2+totw_conf**2)


var2e_conf=ts_mean_conf(var2e)
var3e_conf=ts_mean_conf(var3e)
var4e_conf=ts_mean_conf(var4e)
var5e_conf=ts_mean_conf(var5e)
var6e_conf=ts_mean_conf(var6e)
var1e_conf=np.sqrt(var2e_conf**2+var3e_conf**2+var4e_conf**2+var5e_conf**2+var6e_conf**2)
tote_conf=ts_mean_conf(tote)
var8e_conf=np.sqrt(var1e_conf**2+tote_conf**2)


ebox_means = [0.,           var2e.mean(), var3e.mean(), var4e.mean(), var5e.mean(), var6e.mean(), var8e.mean(),tote.mean()]
wbox_means = [var7w.mean(), var2w.mean(), var3w.mean(), var4w.mean(), var5w.mean(), var6w.mean(), var8w.mean(),totw.mean()]
ebox_conf  = [0.,         var2e_conf, var3e_conf, var4e_conf, var5e_conf, var6e_conf, var8e_conf,0]
wbox_conf  = [var7w_conf, var2w_conf, var3w_conf, var4w_conf, var5w_conf, var6w_conf, var8w_conf,0]


rects3 = ax.bar(x - width*(1./2.), wbox_means, width, yerr=wbox_conf, label='Wbox %s'%model, color='C1', hatch="///")
rects4 = ax.bar(x + width*(1+1./2.), ebox_means, width, yerr=ebox_conf, label='Ebox %s'%model, color='C0', hatch="///")



# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Heat budget (PW)',fontsize=13)
ax.set_title(' ',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([-2,2])
ax.tick_params(axis='x',labelsize=13,rotation=0)
ax.tick_params(axis='y',labelsize=13,rotation=0)
ax.legend(frameon=False,loc='upper left')




########################################################################################################
fig.text(-0.1,-0.05,'b',size=30)
ax = fig.add_axes([0,-0.2,1.,0.15])

model = 'JRA'
box = 'wbox'

var2w = -(current_mlist[model]['180']['%s_euc'%(box)])

var3w = -(current_mlist[model]['180']['%s_necc'%(box)])

var4w = -(current_mlist[model]['180']['%s_secc'%(box)])

var5w = -(current_mlist[model]['180']['%s_nec'%(box)])

var6w = -(current_mlist[model]['180']['%s_sec'%(box)])

var1w = var2w+var3w+var4w+var5w+var6w


box = 'ebox'

var2e = (current_mlist[model]['180']['%s_euc'%(box)])

var3e = (current_mlist[model]['180']['%s_necc'%(box)])

var4e = (current_mlist[model]['180']['%s_secc'%(box)])

var5e = (current_mlist[model]['180']['%s_nec'%(box)])

var6e = (current_mlist[model]['180']['%s_sec'%(box)])

var1e = var2e+var3e+var4e+var5e+var6e


var2w_conf=ts_mean_conf(var2w)
var3w_conf=ts_mean_conf(var3w)
var4w_conf=ts_mean_conf(var4w)
var5w_conf=ts_mean_conf(var5w)
var6w_conf=ts_mean_conf(var6w)
var1w_conf=np.sqrt(var2w_conf**2+var3w_conf**2+var4w_conf**2+var5w_conf**2+var6w_conf**2)

var2e_conf=ts_mean_conf(var2e)
var3e_conf=ts_mean_conf(var3e)
var4e_conf=ts_mean_conf(var4e)
var5e_conf=ts_mean_conf(var5e)
var6e_conf=ts_mean_conf(var6e)
var1e_conf=np.sqrt(var2e_conf**2+var3e_conf**2+var4e_conf**2+var5e_conf**2+var6e_conf**2)

labels = ['EUC','NECC', 'SECC','NEC','SEC']
ebox_means = [var2e.mean(), var3e.mean(), var4e.mean(), var5e.mean(), var6e.mean()]
wbox_means = [var2w.mean(), var3w.mean(), var4w.mean(), var5w.mean(), var6w.mean()]
ebox_conf  = [var2e_conf, var3e_conf, var4e_conf, var5e_conf, var6e_conf]
wbox_conf  = [var2w_conf, var3w_conf, var4w_conf, var5w_conf, var6w_conf]

x = np.arange(len(labels))  # the label locations
width = 0.2                # the width of the bars

rects2 = ax.bar(x - width*(1.+1./2.), wbox_means, width, yerr=wbox_conf, label='Wbox %s55-do'%model, color='C1')
rects1 = ax.bar(x + width*(1./2.), ebox_means, width, yerr=ebox_conf, label='Ebox %s55-do'%model, color='C0')

##########################################################
model = 'CORE'
box = 'wbox'

var2w = -(current_mlist[model]['180']['%s_euc'%(box)])

var3w = -(current_mlist[model]['180']['%s_necc'%(box)])

var4w = -(current_mlist[model]['180']['%s_secc'%(box)])

var5w = -(current_mlist[model]['180']['%s_nec'%(box)])

var6w = -(current_mlist[model]['180']['%s_sec'%(box)])

var1w = var2w+var3w+var4w+var5w+var6w


box = 'ebox'

var2e = (current_mlist[model]['180']['%s_euc'%(box)])

var3e = (current_mlist[model]['180']['%s_necc'%(box)])

var4e = (current_mlist[model]['180']['%s_secc'%(box)])

var5e = (current_mlist[model]['180']['%s_nec'%(box)])

var6e = (current_mlist[model]['180']['%s_sec'%(box)])

var1e = var2e+var3e+var4e+var5e+var6e


var2w_conf=ts_mean_conf(var2w)
var3w_conf=ts_mean_conf(var3w)
var4w_conf=ts_mean_conf(var4w)
var5w_conf=ts_mean_conf(var5w)
var6w_conf=ts_mean_conf(var6w)
var1w_conf=np.sqrt(var2w_conf**2+var3w_conf**2+var4w_conf**2+var5w_conf**2+var6w_conf**2)

var2e_conf=ts_mean_conf(var2e)
var3e_conf=ts_mean_conf(var3e)
var4e_conf=ts_mean_conf(var4e)
var5e_conf=ts_mean_conf(var5e)
var6e_conf=ts_mean_conf(var6e)
var1e_conf=np.sqrt(var2e_conf**2+var3e_conf**2+var4e_conf**2+var5e_conf**2+var6e_conf**2)


ebox_means = [var2e.mean(), var3e.mean(), var4e.mean(), var5e.mean(), var6e.mean()]
wbox_means = [var2w.mean(), var3w.mean(), var4w.mean(), var5w.mean(), var6w.mean()]
ebox_conf  = [var2e_conf, var3e_conf, var4e_conf, var5e_conf, var6e_conf]
wbox_conf  = [var2w_conf, var3w_conf, var4w_conf, var5w_conf, var6w_conf]


rects3 = ax.bar(x - width*(1./2.), wbox_means, width, yerr=wbox_conf, label='Wbox %s'%model, color='C1',hatch='///')
rects4 = ax.bar(x + width*(1+1./2.), ebox_means, width, yerr=ebox_conf, label='Ebox %s'%model, color='C0',hatch='///')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Heat budget (PW)',fontsize=13)
ax.set_title(' ',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([-1.5,1.5])
ax.tick_params(axis='x',labelsize=13,rotation=0)
ax.tick_params(axis='y',labelsize=13,rotation=0)
ax.legend(frameon=False,loc='upper left')

########################################################################################################
fig.text(-0.1,-0.25,'c',size=30)
ax = fig.add_axes([0,-0.4,1.,0.15])

model = 'JRA'

var2w = (current_mlist[model]['180']['euc_Sv'])

var3w = (current_mlist[model]['180']['necc_Sv'])

var4w = (current_mlist[model]['180']['secc_Sv'])

var5w = (current_mlist[model]['180']['nec_Sv'])

var6w = (current_mlist[model]['180']['sec_Sv'])

var1w = var2w+var3w+var4w+var5w+var6w


model = 'CORE'

var2e = (current_mlist[model]['180']['euc_Sv'])

var3e = (current_mlist[model]['180']['necc_Sv'])

var4e = (current_mlist[model]['180']['secc_Sv'])

var5e = (current_mlist[model]['180']['nec_Sv'])

var6e = (current_mlist[model]['180']['sec_Sv'])

var1e = var2e+var3e+var4e+var5e+var6e


var2w_conf=ts_mean_conf(var2w)
var3w_conf=ts_mean_conf(var3w)
var4w_conf=ts_mean_conf(var4w)
var5w_conf=ts_mean_conf(var5w)
var6w_conf=ts_mean_conf(var6w)
var1w_conf=np.sqrt(var2w_conf**2+var3w_conf**2+var4w_conf**2+var5w_conf**2+var6w_conf**2)

var2e_conf=ts_mean_conf(var2e)
var3e_conf=ts_mean_conf(var3e)
var4e_conf=ts_mean_conf(var4e)
var5e_conf=ts_mean_conf(var5e)
var6e_conf=ts_mean_conf(var6e)
var1e_conf=np.sqrt(var2e_conf**2+var3e_conf**2+var4e_conf**2+var5e_conf**2+var6e_conf**2)

labels = ['EUC','NECC', 'SECC','NEC','SEC']
ebox_means = [var2e.mean(), var3e.mean(), var4e.mean(), var5e.mean(), var6e.mean()]
wbox_means = [var2w.mean(), var3w.mean(), var4w.mean(), var5w.mean(), var6w.mean()]
ebox_conf  = [var2e_conf, var3e_conf, var4e_conf, var5e_conf, var6e_conf]
wbox_conf  = [var2w_conf, var3w_conf, var4w_conf, var5w_conf, var6w_conf]


x = np.arange(len(labels))  # the label locations
width = 0.45                # the width of the bars

rects2 = ax.bar(x - width/2, wbox_means, width, yerr=wbox_conf, label='JRA55-do', color='gray')
rects1 = ax.bar(x + width/2, ebox_means, width, yerr=ebox_conf, label='CORE', color='gray',hatch='///')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Volume budget (Sv)',fontsize=13)
ax.set_title('',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([-55,55])
ax.tick_params(axis='x',labelsize=13,rotation=0)
ax.tick_params(axis='y',labelsize=13,rotation=0)
ax.legend(frameon=False,loc='upper right')

fig.tight_layout()

fig.savefig('./figure/figure06abc.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)


########################################################################################################
# # Seasonal amplitude

from time_series_analyses import ts_mean_conf
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,20))
############################################################
fig.text(-0.1,0.15,'a',size=30)
ax = fig.add_axes([0,0,1.,0.15])

model = 'JRA'
box = 'wbox'

var2w = -(current_season_mlist[model]['180']['%s_euc'%(box)])

var3w = -(current_season_mlist[model]['180']['%s_necc'%(box)])

var4w = -(current_season_mlist[model]['180']['%s_secc'%(box)])

var5w = -(current_season_mlist[model]['180']['%s_nec'%(box)])

var6w = -(current_season_mlist[model]['180']['%s_sec'%(box)])

var1w = var2w+var3w+var4w+var5w+var6w


box = 'ebox'

var2e = (current_season_mlist[model]['180']['%s_euc'%(box)])

var3e = (current_season_mlist[model]['180']['%s_necc'%(box)])

var4e = (current_season_mlist[model]['180']['%s_secc'%(box)])

var5e = (current_season_mlist[model]['180']['%s_nec'%(box)])

var6e = (current_season_mlist[model]['180']['%s_sec'%(box)])

var1e = var2e+var3e+var4e+var5e+var6e


labels = ['EUC','NECC', 'SECC','NEC','SEC']
ebox_amps = [var2e.max(), var3e.max(), var4e.max(), var5e.max(), var6e.max()]
wbox_amps = [var2w.max(), var3w.max(), var4w.max(), var5w.max(), var6w.max()]
ebox_phases = [var2e.argmax(), var3e.argmax(), var4e.argmax(), var5e.argmax(), var6e.argmax()]
wbox_phases = [var2w.argmax(), var3w.argmax(), var4w.argmax(), var5w.argmax(), var6w.argmax()]

x = np.arange(len(labels))  # the label locations
width = 0.2                # the width of the bars

rects1 = ax.bar(x + width*(1./2.), ebox_amps, width, label='Ebox %s55-do'%model, color='C0')
rects2 = ax.bar(x - width*(1.+1./2.), wbox_amps, width,  label='Wbox %s55-do'%model, color='C1')

def autolabel(rects,amps,phases):
    """Attach a text label above each bar in *rects*, displaying its height."""

    for nrec,rect in enumerate(rects):
        if amps[nrec]==0:
            continue
        sign=np.float(amps[nrec])/np.abs(amps[nrec])
        if sign > 0 :
            ypos=0.1
        else :
            ypos=-0.5

#         print(sign)

        ax.annotate('%0.0i'%(np.int(phases[nrec])+1),
                    xy=(rect.get_x() + rect.get_width() / 2, amps[nrec]),
                    xycoords='data',
                    xytext=(rect.get_x() + rect.get_width() / 2, amps[nrec]+ypos),
                    textcoords='data',
                    ha='center', va='bottom', rotation=45,size=12)

autolabel(rects1,ebox_amps,ebox_phases)
autolabel(rects2,wbox_amps,wbox_phases)

########################
model = 'CORE'
box = 'wbox'

var2w = -(current_season_mlist[model]['180']['%s_euc'%(box)])

var3w = -(current_season_mlist[model]['180']['%s_necc'%(box)])

var4w = -(current_season_mlist[model]['180']['%s_secc'%(box)])

var5w = -(current_season_mlist[model]['180']['%s_nec'%(box)])

var6w = -(current_season_mlist[model]['180']['%s_sec'%(box)])

var1w = var2w+var3w+var4w+var5w+var6w


box = 'ebox'

var2e = (current_season_mlist[model]['180']['%s_euc'%(box)])

var3e = (current_season_mlist[model]['180']['%s_necc'%(box)])

var4e = (current_season_mlist[model]['180']['%s_secc'%(box)])

var5e = (current_season_mlist[model]['180']['%s_nec'%(box)])

var6e = (current_season_mlist[model]['180']['%s_sec'%(box)])

var1e = var2e+var3e+var4e+var5e+var6e


ebox_amps = [var2e.max(), var3e.max(), var4e.max(), var5e.max(), var6e.max()]
wbox_amps = [var2w.max(), var3w.max(), var4w.max(), var5w.max(), var6w.max()]
ebox_phases = [var2e.argmax(), var3e.argmax(), var4e.argmax(), var5e.argmax(), var6e.argmax()]
wbox_phases = [var2w.argmax(), var3w.argmax(), var4w.argmax(), var5w.argmax(), var6w.argmax()]


rects3 = ax.bar(x + width*(1+1./2.), ebox_amps, width, label='Ebox %s'%model, color='C0',hatch='///')
rects4 = ax.bar(x - width*(1./2.), wbox_amps, width, label='Wbox %s'%model, color='C1',hatch='///')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Heat budget (PW)',fontsize=13)
ax.set_title(' ',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0,1.5])
ax.tick_params(axis='x',labelsize=13,rotation=0)
ax.tick_params(axis='y',labelsize=13,rotation=0)
ax.legend(frameon=False,loc='upper left')


autolabel(rects3,ebox_amps,ebox_phases)
autolabel(rects4,wbox_amps,wbox_phases)


#####################################################################################
fig.text(-0.1,-0.05,'b',size=30)
ax = fig.add_axes([0,-0.2,1.,0.15])

model = 'JRA'

var2w = (current_season_mlist[model]['180']['euc_Sv'])

var3w = (current_season_mlist[model]['180']['necc_Sv'])

var4w = (current_season_mlist[model]['180']['secc_Sv'])

var5w = -(current_season_mlist[model]['180']['nec_Sv'])

var6w = -(current_season_mlist[model]['180']['sec_Sv'])

var7w = -(current_season_mlist[model]['180']['nsec_Sv'])

var1w = var2w+var3w+var4w+var5w+var6w


model = 'CORE'

var2e = (current_season_mlist[model]['180']['euc_Sv'])

var3e = (current_season_mlist[model]['180']['necc_Sv'])

var4e = (current_season_mlist[model]['180']['secc_Sv'])

var5e = -(current_season_mlist[model]['180']['nec_Sv'])

var6e = -(current_season_mlist[model]['180']['sec_Sv'])

var7e = -(current_season_mlist[model]['180']['nsec_Sv'])

var1e = var2e+var3e+var4e+var5e+var6e


labels = ['EUC','NECC', 'SECC','NEC','SEC']

ebox_amps = [var2e.max(), var3e.max(), var4e.max(), var5e.max(), var6e.max()]
wbox_amps = [var2w.max(), var3w.max(), var4w.max(), var5w.max(), var6w.max()]

ebox_phases = [var2e.argmax(), var3e.argmax(), var4e.argmax(),
               var5e.argmax(), var6e.argmax()]
wbox_phases = [var2w.argmax(), var3w.argmax(), var4w.argmax(),
               var5w.argmax(), var6w.argmax()]


x = np.arange(len(labels))  # the label locations
width = 0.45                # the width of the bars

rects2 = ax.bar(x - width/2, wbox_amps, width, label='JRA55-do', color='gray')
rects1 = ax.bar(x + width/2, ebox_amps, width, label='CORE', color='gray',hatch='///')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Volume budget (Sv)',fontsize=13)
ax.set_title('',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0,20])
ax.tick_params(axis='x',labelsize=13,rotation=0)
ax.tick_params(axis='y',labelsize=13,rotation=0)
ax.legend(frameon=False,loc='upper right')



def autolabel(rects,amps,phases):
    """Attach a text label above each bar in *rects*, displaying its height."""

    for nrec,rect in enumerate(rects):
        if amps[nrec]==0:
            continue
        sign=np.float(amps[nrec])/np.abs(amps[nrec])
        if sign > 0 :
            ypos=0.1
        else :
            ypos=-0.5

#         print(sign)

        ax.annotate('%0.0i'%(np.int(phases[nrec])+1),
                    xy=(rect.get_x() + rect.get_width() / 2, amps[nrec]),
                    xycoords='data',
                    xytext=(rect.get_x() + rect.get_width() / 2, amps[nrec]+ypos),
                    textcoords='data',
                    ha='center', va='bottom', rotation=45,size=12)



autolabel(rects2,wbox_amps,wbox_phases)
autolabel(rects1,ebox_amps,ebox_phases)
######################################
fig.text(-0.1,-0.23,'c',size=30)
ax1 = fig.add_axes([0,-0.35,0.15,0.08])
(var2w+current_mlist['JRA']['180']['euc_Sv'].mean()).plot(ax=ax1,color='k',linewidth=2.0,label='JRA55-do')
(var2e+current_mlist['CORE']['180']['euc_Sv'].mean()).plot(ax=ax1,color='k',linestyle='dashed',linewidth=2.0,label='CORE')
ax1.set_title('EUC')
ax1.set_ylabel('Sv',{'size':'12'})
ax1.tick_params(axis='x',rotation=90)
ax1.set_xticks(np.arange(12)+1)
ax1.legend(frameon=False,loc='upper right',bbox_to_anchor=(0.3, 1.5))

ax1 = fig.add_axes([0.21,-0.35,0.15,0.08])
(var3w+current_mlist['JRA']['180']['necc_Sv'].mean()).plot(ax=ax1,color='k',linewidth=2.0)
(var3e+current_mlist['CORE']['180']['necc_Sv'].mean()).plot(ax=ax1,color='k',linestyle='dashed',linewidth=2.0)
ax1.set_title('NECC')
ax1.set_ylabel('',{'size':'12'})
ax1.tick_params(axis='x',rotation=90)
ax1.set_xticks(np.arange(12)+1)

ax1 = fig.add_axes([0.42,-0.35,0.15,0.08])
(var4w+current_mlist['JRA']['180']['secc_Sv'].mean()).plot(ax=ax1,color='k',linewidth=2.0,label='JRA55-do')
(var4e+current_mlist['CORE']['180']['secc_Sv'].mean()).plot(ax=ax1,color='k',linestyle='dashed',linewidth=2.0,label='CORE')
ax1.set_title('SECC')
ax1.set_ylabel('',{'size':'12'})
ax1.tick_params(axis='x',rotation=90)
ax1.set_xticks(np.arange(12)+1)

ax1 = fig.add_axes([0.63,-0.35,0.15,0.08])
(var5w+current_mlist['JRA']['180']['nec_Sv'].mean()).plot(ax=ax1,color='k',linewidth=2.0)
(var5e+current_mlist['CORE']['180']['nec_Sv'].mean()).plot(ax=ax1,color='k',linestyle='dashed',linewidth=2.0)
ax1.set_title('NEC')
ax1.set_ylabel('',{'size':'12'})
ax1.tick_params(axis='x',rotation=90)
ax1.set_xticks(np.arange(12)+1)

ax1 = fig.add_axes([0.84,-0.35,0.15,0.08])
(var6w+current_mlist['JRA']['180']['sec_Sv'].mean()).plot(ax=ax1,color='k',linewidth=2.0)
(var6e+current_mlist['CORE']['180']['sec_Sv'].mean()).plot(ax=ax1,color='k',linestyle='dashed',linewidth=2.0)
# var7w.plot(ax=ax1,color='C0',linestyle='dashed',linewidth=2.0)
# var7e.plot(ax=ax1,color='C1',linestyle='dashed',linewidth=2.0)
ax1.set_title('SEC')
ax1.set_ylabel('',{'size':'12'})
ax1.tick_params(axis='x',rotation=90)
ax1.set_xticks(np.arange(12)+1)


fig.tight_layout()

fig.savefig('./figure/figure19.pdf', dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=None,
                frameon=None)
