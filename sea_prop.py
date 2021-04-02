#!python

import numpy as np 


"""
The module is written at UCI in Velicogna Lab
further modified to save memories and compatible 
with xarray 


The module is written by Chia-Wei Hsu for calculating 
various sea level related properties including

    1. ocean mass  (rho with depth provided, ssh dataset required)
    2. steric height (rho with pressure level provided)
    3. ocean density profile (defined by the Joint Panel on Oceanograhic Tables and Standards)
        this density profile derivation is old and not satifying the TOES10 thermodynamic
        constrain. RECOMMEND using gsw package (https://teos-10.github.io/GSW-Python/intro.html)
        
        
updates:
- 

    
"""



#==========================================================================
# Ocean mass of the ocean 
#   use intergration over the entire column of water 
#   with the density profile to calculate the ocean mass 
#   (not OBP since it does not include the atmospheric pressure)
# 
# Input :
#   rho     (kg/m^3)
#   ssh     (m)
#   dz      (m)
#   outunit (option: cmH2O, mbar) output omass unit
#

def ocean_mass(rho,ssh,dz,outunit='cmH2O'):
    g=np.float64(9.81)             # m/s^2
    rhow=np.float(1000.)           # kg/m^3
    rhosw=np.float(1030.)          # kg/m^3 (sea water)

    omp=np.nansum(rho*dz, axis=2)*g    # kg/m/s^2 (Pa)
    omp=omp+ssh*rhosw*g                

    if outunit in ['cmH2O']:
       h=omp/rhow/g*100.               # kg/m/s^2 * m^3/kg * s^2/m * cm/m
    if outunit in ['mbar']:
       h=omp/100.                      # Pa/100 = mbar

    return h




#==========================================================================
# Steric height of the ocean 
#   use hydrostatic balance (PGF + gravity = 0)
#   steric height = dynamic height / g 
#   methods:
#     D(z1,z2) = g*(z1-z2) = -sum_i(1/rho(i)*dp(i)) 
#     H = z1-z2 = {-sum_i[dp(i)/rho(i)]}/g 
#   Input:
#     IMPORTANT !!!!!! THE INPUT ARRAY LISTED DELOW HAS TO BE IN THREE DIMENSION [NLON,NLAT,LEVEL]
#                      NO MORE NO LESS
#     dp  (Pa)
#     rho (kg/m^3)
#   Output: 
#     H   (m)
#
def steric_hgt(rho,dz):
    #constant
    g=np.float64(-9.81)
    H=-np.nansum(dp/rho)/g 
    return H

#==========================================================================
# Thermosteric height of the ocean 
#   use hydrostatic balance (PGF + gravity = 0)
#   steric height = dynamic height / g 
#   methods:
#     D(z1,z2) = g*(z1-z2) = -sum_i(1/rho(i)*dp(i)) 
#     H = z1-z2 = {-sum_i[dp(i)/rho(i)]}/g 
#   Input:
#     IMPORTANT !!!!!! THE INPUT ARRAY LISTED DELOW HAS TO BE IN THREE DIMENSION [NLON,NLAT,LEVEL]
#                      NO MORE NO LESS
#     dp  (Pa)
#     rho (kg/m^3)
#   Output: 
#     H   (m)
#
def thermosteric_hgt(alphadT,dz):
    #constant
    dH=np.nansum(alphadT*dz)
    return dH



#------
# The following equation is defined by the Joint Panel on Oceanograhic Tables and Standards 
#  fits avaiable measurements with a standard error of 3.5 ppm 
#  for pressure up to 1000bar (10000dbar)
#  for temperatures between freezing and 40 degree C
#  for salinities between 0 and 42 PSU
#  density rho (kg/m^3) is expressed in terms of 
#     - pressure p (bars) 
#     - temperature (degree C)
#     - practical density (PSU=PSS-78: approx to parts per thousand)
#------
#==========================================================================
# Density of sea water 
#   equation provided from Gill appendix 
#   Input:
#     pressure    - P(bar)
#     temperature - T(degree C)  
#     salinity    - S(PSU)
#   Output: 
#     density     - rho(kg/m^3)
 
def density_sea(S,T,P):
#     print '--calculate the density in the ocean'
    dict1 = density_sea_1stdatm(S,T)
    rho_sea_1stdatm = dict1['density']
    dict2 = K_sea(S,T,P)
    K_s = dict2['K']
    rho_sea = rho_sea_1stdatm / (1-P/K_s) 
    return{'density':rho_sea}
        
#==========================================================================
# Density of sea water at one standard atmosphere (p=0 => ocean surface) 
#   equation provided from Gill appendix 
#    
def density_sea_1stdatm(S,T):
    dict1 = density_water(T)
    rho_w = dict1['density']
    rho_sea_1stdatm = rho_w + S*(0.824493-4.0899e-3*T+7.6438e-5*T**2-8.2467e-7*T**3+5.3875e-9*T**4)\
                      + S**(1.5)*(-5.72466e-3+1.0227e-4*T-1.6546e-6*T**2) + S**2*(4.8314e-4)
    return{'density':rho_sea_1stdatm}

#==========================================================================
# Density of pure water 
#   equation provided from Gill appendix 
#    
def density_water(T):
    rho_w = 999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5
    return{'density':rho_w}


#==========================================================================
# Compressibility of sea water   
#   equation provided from Gill appendix 
#    
def K_sea(S,T,P):
    dict1 = K_sea_1stdatm(S,T)
    K_s_1stdatm = dict1['K']
    K_s = K_s_1stdatm + P*(3.239908+1.43713e-3*T+1.16092e-4*T**2-5.77905e-7*T**3)\
            + P*S*(2.2838e-3-1.0981e-5*T-1.6078e-6*T**2) + 1.91075e-4*P*S**(1.5)\
            + P**2*(8.50935e-5-6.12293e-6*T+5.2787e-8*T**2)\
            + P**2*S*(-9.9348e-7+2.0816e-8*T+9.1697e-10*T**2)
    return{'K':K_s}

#==========================================================================
# Compressibility of sea water at one standard atmosphere (p=0 => ocean surface)  
#   equation provided from Gill appendix 
#    
def K_sea_1stdatm(S,T):
    dict1 = K_water(T)
    K_w = dict1['K']
    K_s_1stdatm = K_w + S*(54.6746-0.603459*T+1.09987e-2*T**2-6.1670e-5*T**3)\
                    + S**(1.5)*(7.944e-2+1.6483e-2*T-5.3009e-4*T**2)
    return{'K':K_s_1stdatm}

#==========================================================================
# Compressibility of pure water (secant bulk modulus)
#   equation provided from Gill appendix 
#    
def K_water(T):
    K_w = 19652.21 + 148.4206 * T - 2.327105 * T**2 + 1.360477e-2 * T**3 - 5.155288e-5 * T**4 
    return{'K':K_w}
