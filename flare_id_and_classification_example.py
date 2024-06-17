#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:31:30 2023

@author: idavis
"""

from flare_id import *
import numpy as np
import lightkurve as lk
from astropy import units as un

#%% Starting from scratch with bare minimum information on a star:
tic_num = 416741712
radius = 0.91 # solar radii
temperature = 5665 # Kelvin

# make Star object:
star = Star(tic_num=tic_num, radius=radius, temperature=temperature)

print(f'Calculated period: {star.period}')
print(f'Sectors the star appears in: {star.lcs.sector}')

# make Flares object from Star object and do processing:
flares = Flares(star)
flares.FlagLightCurves()

print(f'Stats: Median = {flares.median}, std= {flares.std}')

flares.FindAllFlares()
flares.ConsolidateComplexFlares()


fig, axs = flares.PlotLightCurves()
for i in range(len(flares.lc_flagged)):
    t = flares.lc_flagged[i].time.value
    f = flares.lc_median[i]
    axs[0].plot(t,f,c = 'r',label = 'Median-filtered light curve')
axs[0].legend()


# change window size to get better median filtered light curve:
flares.window = 175
flares.FlagLightCurves()
print(f'Stats after new window size: Median = {flares.median}, std= {flares.std}')

flares.FindAllFlares()
flares.ConsolidateComplexFlares()
print(f'Flare rate: {flares.flare_rate}')

fig, axs = flares.PlotLightCurves()
for i in range(len(flares.lc_flagged)):
    t = flares.lc_flagged[i].time.value
    f = flares.lc_median[i]
    axs[0].plot(t,f,c = 'r',label = 'Median-filtered light curve')
axs[0].legend()


# flag flares that don't meet a minimum energy threshold:
E_min = 5e34
inds = np.where(flares.flare_table['energy'] < E_min)[0]
flares.FlagFlares(inds)
flares.PlotLightCurves()

print(f'Flare rate after flagging: {flares.flare_rate}')
print(flares.flare_table)

# write out data; default is to save to the directory you're working in:
flares.WriteOutData()

#%% Starting from already having a light curve collection + star information:
    
# get some example data and use a different integration time:
lcs = lk.search_lightcurve(f'TIC {tic_num}', author = 'SPOC',exptime = 120).download_all()
for i in range(len(lcs)):
    lc = lcs[i].bin(5 * un.min)
    lcs[i] = lc

star = Star(tic_num=tic_num, radius=radius, temperature=temperature, lcs=lcs)
flares = Flares(star, int_time=300 * un.s)
flares.FlagLightCurves()
flares.FindAllFlares(n_points_max=2)
flares.PlotLightCurves()
