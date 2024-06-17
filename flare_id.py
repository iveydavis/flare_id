#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:46:07 2023

@author: idavis
"""

import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
from astropy import units as un, constants as const
import glob
from scipy.signal import medfilt
from scipy.integrate import quad
import os
import json
import pandas as pd
from warnings import warn
import shutil


cache_dir = None
try:
    cache_dir = lk.config.get_cache_dir()
except:
    conf = lk.config.get_config_dir()
    cache_dir = f'{conf.rstrip("/config")}-cache'
    warn("Could not get the cache directory. Assuming legacy location {cache_dir} ")

class Star:
    """
    Class for holding information about a star observed by TESS
    """
    def __init__(self,tic_num:int, radius:'Solar radii', temperature:'Kelvin', lcs:lk.LightCurveCollection = None, period = None, sectors = range(14), exp_time:'sec' = 120, clear_cache=True, cache=cache_dir):
        f"""
        :param tic_num: TIC number of the object
        :type tic_num: int
        :param radius: Radius of the star in solar radii
        :type radius: float
        :param temperature: Temperature of the star in Kelvin
        :type temperature: float
        :param lcs: collection of light curves associated with the star. If None, then the class will search for it with lightkurve, defaults to None
        :type lcs: lk.LightCurveCollection, optional
        :param period: Period of the stars in days. If None, then will do BLS Lomb Scargle to determine it, defaults to None
        :type period: float, optional
        :param sectors: Which TESS sectors to collect light curves from if lcs is None, defaults to range(14)
        :type sectors: list of integers, optional
        :param exp_time: exposure time of light curves to search for, defaults to 120 s
        :type exp_time: int
        :param clear_cache: Boolean telling whether the cache directory of the lightkurve for the star should be deleted. Default is True
        :type clear_cache: bool
        :param cache: the cache directory. Default is {cache_dir}
        :type cache: str
        """
        assert(float(radius))
        assert(float(temperature))
        if period is not None:
            assert(float(period))
        
        self.tic_num = tic_num
        self.radius = radius
        self.temperature = temperature
        self.cache_dir = cache
        
        # Get light curves from TESS sectors:
        if lcs is None:
            lcs = lk.search_lightcurve(f'TIC {tic_num}', exptime=exp_time, author='SPOC', sector=sectors).download_all()
            lc = []
            for l in lcs:
                if int(l.meta['LABEL'].strip('TIC ')) == tic_num:
                    lc.append(l)
            self.lcs = lk.LightCurveCollection(lc)
                    
        elif lcs is not None:
            self.lcs = lcs
            try:
                self.lcs[0].flux.mask
            except:
                for i in range(len(lcs)):
                    l = lcs[i]
                    mask = np.isnan(l.flux)
                    l.flux = lk.LightCurve.MaskedColumn(data = l.flux,name = 'flux', mask = mask)
            
        # Calculate the average period between the different sector light curves using BLS method:
        if period is None:
            periods = []
            mps = []
            for lc in self.lcs:
                p = lc.to_periodogram('bls')
                periods.append(p.period_at_max_power.to('day').value)
                mps.append(p.max_power.value)
            mp = np.nanmean(mps)
            if mp >= 1e3:
                period = np.average(periods)
            elif mp < 1e3:
                period = np.nanmean([len(lc) for lc in lcs]) * exp_time * un.s.to('d')
            self.period = period
            
        elif period is not None:
            assert(float(period))
            self.period = period
            
        if clear_cache:
            self.clear_cache()
        return
    
    def clear_cache(self):
        for lc in self.lcs:
            fn = lc.meta['FILENAME']
            path = os.path.dirname(fn)
            if os.path.isdir(path):
                try:
                    assert(self.cache_dir in fn), f"File {fn} is not in the cache directory {cache_dir}."
                    shutil.rmtree(path)
                except:
                    warn(f"Could not remove directory {fn}")
    
class Flares:
    """
    Class for flagging flares from a star observed by TESS and storing that data
    """
    def __init__(self,star:Star, process:bool = True,int_time = 120*un.s):
        """
        :param star: Star object to do the processing on
        :type star: Star
        :param process: Tells whether it should calculate the window sizes and light curve array split by gaps in data, defaults to True
        :type process: bool, optional
        :param int_time: integration time of the light curves, defaults to 120s
        :type int_time: astropy.units.quantity.Quantity
        """
        
        assert(int_time.unit.to('min')),"The integration time needs to be in units of time"
        self.star = star
        self.int_time = int_time
        
        if process:
            #Identify where there are flagged points and remove them from the light curve:
            lc = star.lcs.stitch()
            keep_inds = np.where(lc.flux.mask == False)[0]
            self.lc = lc[keep_inds]
            
            #Determine the window time as 1/15 the rotation period:
            window_time = (self.star.period*1440/15) #convert to minutes
            #Assign minimum window time
            if window_time < 100:
                window_time = 100
            
            #Convert window time to 
            window = int((window_time/int_time.to('min').value))
            
            #Make the window size odd:
            if window%2 == 0:
                window -= 1
                
            self.window = window
            self.eclipse_window = 2 * window -1
            
            #Split the light curve into an array of light curves separated by excessive gaps in time:
            
        elif not process:
            self.window = None
            self.eclipse_window = None
            self.lc_arr = None
            self.lc = None
        
        #Intialize all the parameters that will be calculated
        self.lc_arr = None
        self.lc_flagged = None
        self.lc_median = None
        self.flares = None
        self.lc_norm = None
        self.std = None
        self.flare_table = None
        self.flare_rate = None
        self._n_pts = 0
        return
    
    def SplitLightCurve(self, min_gap = 30 * un.min):
        """
        Splits the light curve into a list of light curves based on gaps in the original curve
        :param min_gap: The minimum gap size for two light curves sections to be considered separate, defaults to 30 * un.min
        :type min_gap: astropy.units.quantity.Quantity, optional
        :return: Updates the lc_arr property
        :rtype: list of LightCurve objects

        """
        lcs = []
        
        #Find difference in time between consecutive points:
        s = pd.Series(self.lc.time.value)
        d = s.diff()
        
        #Identify where gaps are greater than the minimum allowed gap
        inds = np.where(d > min_gap.to('d').value)[0]
        
        #Go through the indices where the gaps occur to split:
        i0 = 0
        for i in range(len(inds)):
            l = self.lc[i0:inds[i]-1]
            i0 = inds[i]
            
            #Exclude sections of the light curve that are shorter than the window size:
            if len(l) > self.window:
                lcs.append(l)
        
        #Get the last chunk of the light curve:
        if len(self.lc[i0:]) > self.window:
            lcs.append(self.lc[i0:])

        self.lc_arr = lcs
    
    def __SplitFlares(self, lc, min_gap = 6 * un.min):
        """
        Splits a lightcurve of flares into individual flares
        :param lc: The light curve of flares to split
        :type lc: LightCurve
        :param min_gap: The maximum allowed time between two points for them to be considered apart of the same flare, defaults to 6 * un.min
        :type min_gap: astropy.units.quantity.Quantity, optional
        :return: array of flare light curves
        :rtype: list of LightCurve objects

        """
        lcs = []
        
        #Find difference in time between consecutive points:
        s = pd.Series(lc.time.value)
        d = s.diff()
        
        #Identify where gaps are greater than the minimum allowed gap
        inds = np.where(d > min_gap.to('d').value)[0]
        
        #Go through the indices where the gaps occur to split:
        i0 = 0
        for i in range(len(inds)):
            l = lc[i0:inds[i]-1]
            i0 = inds[i]
            lcs.append(l)
            
        #Get the last chunk of the light curve:
        lcs.append(lc[i0:])
        
        return lcs

    
    def __FlagPoints(self,lc:lk.LightCurve, max_iter:int = 20):
        """
        Iteratively flags points in excess of 3 sigma of the median of the normalized light curve
        :param lc: The light curve to be flagged
        :type lc: lk.LightCurve
        :param max_iter: Number of iterations to try, defaults to 20
        :type max_iter: int, optional
        :return: The flagged light curve
        :rtype: lk.LightCurve

        """
        #Make copy of light curve to edit
        lc_new = lk.LightCurve(lc)
        flux = lc_new.flux
        
        #initialize list of index values for anomalous points and counter for number of iterations
        flag_inds = [0]
        max_iter_count = 0
        
        while len(flag_inds) != 0 and max_iter_count <= max_iter:
            #Normalize the lightcurve by the median-filtered light curve:
            lc_medfilt = medfilt(flux, self.window)
            lc_norm = flux/lc_medfilt
            
            #Identify indices in greater than 3 sigma of the normalized light curve:
            sig = np.nanstd(lc_norm)
            med = np.nanmedian(lc_norm)
            flag_inds = np.where(lc_norm >= med + 3 *sig)[0]
            
            #redo normalization with the eclipse window instead of window:
            lc_medfilt = medfilt(flux, self.eclipse_window)
            lc_norm = flux/lc_medfilt
            
            #Identify indices in less than than 3 sigma of the normalized light curve:
            sig = np.std(lc_norm)
            med = np.median(lc_norm)
            eclipse_inds = np.where(lc_norm <= med - 3*sig)[0]
            flag_inds = np.concatenate((flag_inds, eclipse_inds))
            
            if len(flag_inds) > 0:
                #flag indices via mask values:
                new_mask = flux.mask
                new_mask[flag_inds] = True
                flux.mask = new_mask
                
            max_iter_count += 1
            
        lc_new.flux = flux
        return lc_new
    
    def FlagLightCurves(self, max_iter:int = 20):
        """
        Flags outlying points in all light curves in the lc_arr property and updates the lc_flagged, lc_median, lc_norm, median, and std properties
        :param max_iter: Maximum number of times to iteratively flag points, defaults to 20
        :type max_iter: int, optional

        """
        if self.lc_arr is None:
            warn("The light curve has not been split into sections yet. They will now be split with the default gap of 30 min.")
            self.SplitLightCurve()
            
        flagged_lcs = []
        lc_medfilts = []
        norm_lc = []
        medians = []
        
        for l in self.lc_arr:
            # Flag points and add them to the final list:
            flagged_lc = self.__FlagPoints(l, max_iter)
            flagged_lcs.append(flagged_lc)
            
            #Calculate the median-filter curve of the flagged light curve and add them to a list:
            med_lc = medfilt(flagged_lc.flux, self.window)
            lc_medfilts.append(med_lc)
            
            #Calculate the normalized, unflagged light curve and add them to a list:
            norm_lc.append(l/med_lc)
            medians.append(np.nanmedian(flagged_lc.flux/med_lc))
            
        self.median = np.nanmedian(medians)
        self.lc_flagged = flagged_lcs
        self.lc_median = lc_medfilts
        self.lc_norm = norm_lc
        self.CalculateSTD()
    
          
    def __FindFlares(self, lc_raw:lk.LightCurve, lc_flagged:lk.LightCurve,clip_size:int=None,sigma_max_threshold = 3, sigma_min_threshold = 2,n_points_max = 3,n_points_min = 3,min_gap = 6 * un.min):
        """
        Finds the flares in a light curve
        :param clip_size: The number of points to clip on the edges of light curves defaults to None
        :type clip_size: int, optional
        :param sigma_max_threshold: upper required threshold to be considered apart of the flare, defaults to 3
        :type sigma_max_threshold: float, optional
        :param sigma_min_threshold: minimum required threshold to be considered apart of the flare, defaults to 2
        :type sigma_min_threshold: float, optional
        :param n_points_max: Minimum number of upper threshold points required to be considered a flare, defaults to 3
        :type n_points_max: float, optional
        :param n_points_min: Minimum number of minimum threshold points required to be considered a flare, defaults to 3
        :type n_points_min: float optional
        :param min_gap: Maximum allowed time between consecutive flagged points to be considered apart of the same flare, defaults to 6 * un.min
        :type min_gap: astropy.units.quantity.Quantity,, optional
        :return: List of flare light curves
        :rtype: List of lk.LightCurve objects

        """
        #Make sure the sigma threshold is a float or an int:
        assert(float(sigma_min_threshold))
        assert(float(sigma_max_threshold))
        
        #Assigns clip_size to the window size if it was set to None:
        if clip_size is None:
            if self.window >= 209:
                clip_size = int(self.window/3)
            else:
                clip_size = 70
            
        #Make copies of light curves to do editing:
        lc_raw_copy = lk.LightCurve(lc_raw)
        lc_flagged_copy = lk.LightCurve(lc_flagged)
        
        #Get normalized light curve, ignoring edges based on clip_size:
        lc_medfilt = medfilt(lc_flagged_copy.flux, self.window)    
        norm_flux = (lc_raw_copy.flux/lc_medfilt)[clip_size:-clip_size]
        
        #Assign normalized, clipped light curve flux to the copied lc_raw:
        lc_raw_copy = lk.LightCurve(lc_raw)[clip_size:-clip_size]
        lc_raw_copy.flux = norm_flux
        
        #Assign normalized, flagged, clipped light curve flux to a variable and do statistics:
        norm_flux_flagged = (lc_flagged_copy.flux/lc_medfilt)[clip_size:-clip_size]
        sig = np.nanstd(norm_flux_flagged)
        med = np.nanmedian(norm_flux_flagged)
        
        #Identify the points in the normalized, clipped light curve that meet the significance threshold:
        flag_inds = np.where(lc_raw_copy.flux >= med + sigma_min_threshold *sig)[0]
        flare_candidates = lc_raw_copy[flag_inds]
        
        #Split flare light curve into individual flares:
        flare_lcs = self.__SplitFlares(flare_candidates,min_gap)
        final_flare_lcs = []
        
        if len(flare_lcs) > 0:
            for i in flare_lcs:
                #Make sure there's at least n points in the light curve:
                if len(i) >= n_points_min:
                    #make sure the correct number of sigma_max and mean sigma points are present:
                    max_sig_inds = np.where(i.flux >= med + sigma_max_threshold * sig)[0]
                    mean_sig_inds = np.where(i.flux >= med + (sigma_max_threshold +sigma_min_threshold)*sig/2)[0]
                    
                    if len(max_sig_inds) >= n_points_max and len(mean_sig_inds) - n_points_max >= n_points_max/2:
                        #Find where the tiems of the flare light curves correspond to the original raw light curve:
                        t0_ind = np.where(lc_raw_copy.time == i.time[0])[0][0]
                        te_ind = np.where(lc_raw_copy.time == i.time[-1])[0][0]
                        final_flare_lcs.append(lk.LightCurve(lc_raw_copy[t0_ind:te_ind+1]))
        n_pts = len(norm_flux)
        return final_flare_lcs,n_pts
    
    
    def FindAllFlares(self, clip_size:int = None, sigma_max_threshold = 3, sigma_min_threshold = 2,n_points_max = 3,n_points_min = 3, min_gap = 6 * un.min,T_flare = 10_000*un.K):
        """
        Finds all the flares in the lc_norm property, calculates flare rate, and makes flare table
        :param clip_size: The number of points to clip on the edges of light curves defaults to None
        :type clip_size: int, optional
        :param sigma_max_threshold: upper required threshold to be considered apart of the flare, defaults to 3
        :type sigma_max_threshold: float, optional
        :param sigma_min_threshold: minimum required threshold to be considered apart of the flare, defaults to 2
        :type sigma_min_threshold: float, optional
        :param n_points_max: Minimum number of upper threshold points required to be considered a flare, defaults to 3
        :type n_points_max: float, optional
        :param n_points_min: Minimum number of minimum threshold points required to be considered a flare, defaults to 3
        :type n_points_min: float optional
        :param min_gap: Maximum allowed time between consecutive flagged points to be considered apart of the same flare, defaults to 6 * un.min
        :type min_gap: astropy.units.quantity.Quantity,, optional
        :param T_flare: Temperature to assume for the flares, defaults to 10_000*un.K
        :type T_flare: astropy.units.quantity.Quantity, optional

        """
        #Make sure that light curves have been flagged:
        if self.lc_flagged is None:
            warn("FlagLightCurves hasn't be run yet. Running it now with default values.")
            self.FlagLightCurves()
            
        #Assigns clip_size to the window size if it was set to None:
        if clip_size is None:
            if self.window >= 209:
                clip_size = int(self.window/3)
            else:
                clip_size = 70
        all_flares = []
        n_pts_tot = 0
        
        #Goes through all light curves in lc_arr to find flares:
        for i in range(len(self.lc_arr)):
            lc_orig = self.lc_arr[i]
            lc_flagged = self.lc_flagged[i]
            flares,n_pts = self.__FindFlares(lc_orig, lc_flagged, clip_size=clip_size,sigma_max_threshold=sigma_max_threshold,sigma_min_threshold = sigma_min_threshold, n_points_max=n_points_max, n_points_min=n_points_min,min_gap = min_gap)
            n_pts_tot += n_pts
            if flares is not None:
                for f in flares:
                    all_flares.append(f)
        
        # Assigns values to class properties:
        self.flares = all_flares
        self._n_pts = n_pts_tot
        self.MakeFlareTable()
        self.CalculateFlareRate()
        
        
        return
    
    def FlagFlares(self, bad_flare_list:list):
        """
        Flags mis-identified flares in the flare table, masks the flare's flux in the flares property, and updates the flare_rate property
        :param bad_flare_list: List of indices of the flares property that have mis-identified flares
        :type bad_flare_list: list

        """
        if self.flares == None:
            raise Warning("There are no flares in the flare property. There are either no flares for this star, or you need to run FindAllFlares")
        for index in bad_flare_list:
            self.flare_table[index]['flag'] = True
            self.flares[index].flux.mask[:] = True
        self.CalculateFlareRate()
        return
    
    def UnflagFlares(self, good_flare_list: list):
        """
        Unflags flagged flares in the flare table, unmasks the flare's flux in the flares property, and updates the flare_rate property
        :param good_flare_list: List of indices of the flares property that need to be unflagged
        :type good_flare_list: list
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if self.flares == None:
            raise Warning("There are no flares in the flare property. There are either no flares for this star, or you need to run FindAllFlares")
        for index in good_flare_list:
            self.flare_table[index]['flag'] = False
            self.flares[index].flux.mask[:] = False
        self.CalculateFlareRate()
        return
    
    def MakeFlareTable(self,T_flare = 10_000*un.K):
        """
        Makes a table of information of the flares. Includes the start time, energy, duration, maximum luminosity, and flag column
        :param T_flare: Temperature of the flare, defaults to 10_000*un.K
        :type T_flare: astropy.units.quantity.Quantity, optional

        """
        integration_time = self.int_time
        if self.flares == None:
            self.FindAllFlares()
            
        flare_table = Table(names=['flare_start','duration','energy','max_L','flag'],dtype=[float,float,float,float,bool])
        
        for f in self.flares:
            flare_start = f.time[0].value
            flare_duration = (f.time[-1] - f.time[0]).to('s')
            
            L = self.CalculateFlareLuminosity(f.flux,T_flare)
            energy = (L * integration_time).to('erg')
            E = np.nansum(energy)
            max_luminosity = np.max(L)
            flare_table.add_row([flare_start,flare_duration,E,max_luminosity,False])
            
        self.flare_table = flare_table
        self.CalculateFlareRate()
        return
    
    
    def ConsolidateComplexFlares(self,max_time_spacing = 2.4 * un.hr):
        """
        Consolidates information for flares that occur close enough together that they might be considered apart of the same complex flare
        :param max_time_spacing: Minimum amount of time between the start of one flare and the end of the next flare for them to be considered separate events, defaults to 2.4 * un.hr
        :type max_time_spacing: astropy.units.quantity.Quantity, optional

        """
        #Makes a table of flares if it isn't already made:
        if self.flare_table == None:
            raise Warning("There is no flare table. Run FindAllFlares")
            
        #Check to make sure there is actually more than one flare detected to consolidate:
        if len(self.flare_table) > 1:
            flare_tab = self.flare_table
            flare_tab.sort('flare_start')
            
            #Convert all flare start times to btjd values:
            flare_starts = Time(flare_tab['flare_start'],format = 'btjd')
            
            #Determines the btjd values for the ends of all flares:
            flare_ends = flare_starts + flare_tab['duration']*un.s
            
            flares = self.flares
            
            #Starts the list of consolidated flares with the first flare
            consolidated_flares = [flares[0]]
            start_ind = 0
            end_ind = start_ind + 1
            
            while end_ind <= len(flares) - 1:
                #Calculate the time difference between the end of the next flare and the beginning of the first flare:
                dt = flare_ends[end_ind] - flare_starts[start_ind]
                
                #If the next flare occurs within max_time_spacing:
                if dt <= max_time_spacing:
                    #Update the flare's light curve to include the next flare's lightcurve:
                    consolidated_flares[-1] = consolidated_flares[-1].append(flares[end_ind])
                    
                    #update start_ind and end_ind to check if the next flare is also within this time range:
                    start_ind += 1
                    end_ind += 1
                
                #If the next flare isn't within the max_time_spacing:
                elif dt > max_time_spacing:
                    #Add the flare as its own element in the consolidated_flare list and update indices:
                    consolidated_flares.append(flares[end_ind])
                    start_ind = end_ind
                    end_ind = start_ind + 1
        
        else:
            print('Not enough flares to consolidate')
            return
        print(f'{len(self.flares)} flares consolidated to {len(consolidated_flares)}')
        #Update flare list and flare table:
        self.flares = consolidated_flares
        self.MakeFlareTable()
        
    def CalculateFlareRate(self):
        """
        Calculates the flare rate in units of per year

        """

        duration = self._n_pts * self.int_time
        n_flares = len(np.where(self.flare_table['flag'] == False)[0])
        flare_rate = (n_flares/duration).to('yr**(-1)')
        self.flare_rate = flare_rate
        return
    
    def PlotLightCurves(self, show_flares:bool = True):
        """
        Plots the light curves!
        :param show_flares: Tells whether to show the location of identified flares, defaults to True
        :type show_flares: bool, optional
        :return: figure and axes of the plot
        :rtype: matplotlib.figure.Figure,list(matplotlib.axes._subplots.AxesSubplot)

        """
        
        # if plot_original and plot_norm:
        fig,ax = plt.subplots(2,1,sharex=True)
        for l in self.lc_arr:
            l.scatter(ax = ax[0],c = 'k')
        for l in self.lc_norm:
            l.scatter(ax = ax[1],c = 'k')
        leg = ax[0].get_legend()
        leg.remove()
        leg = ax[1].get_legend()
        leg.remove()
        y_top1,y_bottom1 = ax[0].get_ybound()
        y_top2,y_bottom2 = ax[1].get_ybound()
        if show_flares == True:
            if self.flare_table == None:
                self.MakeFlareTable()
            for i in range(len(self.flare_table)):
                f = self.flare_table[i]
                if f['flag'] == False:
                    t0 = f['flare_start']
                    ax[0].plot(np.linspace(t0,t0),np.linspace(y_top1,y_bottom1),c = 'r',alpha = 0.4)
                    ax[1].plot(np.linspace(t0,t0),np.linspace(y_top2,y_bottom2),c = 'r',alpha = 0.4)
                    self.flares[i].scatter(ax = ax[1], c= 'r')
            leg = ax[1].get_legend()
            try:
                leg.remove()
            except:
                pass
        ax[0].set_ylim(y_top1,y_bottom1)
        ax[1].set_ylim(y_top2,y_bottom2)
        plt.tight_layout()
        return fig,ax
    
    def CalculateFlareLuminosity(self,flare_lc,T_flare = 10_000*un.K):
        """
        Calculates the bolometric luminosity of a point in a light curve
        :param flare_lc: The point in the DETRENDED flare light curve to calculate the luminosity of
        :type flare_lc: lk.LightCurve
        :param T_flare: Temperature of the flare, defaults to 10_000*un.K
        :type T_flare: astropy.units.quantity.Quantity, optional
        :return: Luminosity of the flare in erg/s
        :rtype: astropy.units.quantity.Quantity

        """
        #Determine difference in flux:
        c = flare_lc - self.median
        
        #Area of the star
        area = 4 * np.pi * (self.star.radius*const.R_sun)**2
        
        #prefactor to fraction in second equation of appendix A:
        prefac = c * area * const.sigma_sb * T_flare**4
        
        #Intensities of the flare and quiescent star 
        I_star = quad(PlanckFunction, 600, 1000,args = self.star.temperature)[0] * un.W/un.m**2
        I_fl = quad(PlanckFunction, 600, 1000,args = T_flare.value)[0]* un.W/un.m**2
        L_fl = (prefac * I_star/I_fl).to('erg/s')
        return L_fl
    
    def CalculateFlareEnergy(self,flare_lc:lk.LightCurve(),T_flare = 10_000*un.K):
        """
        Calculates the bolometric energy of the entire flare.
        :param flare_lc: The light curve of the detrended flare event
        :type flare_lc: lk.LightCurve()
        :param T_flare: Temperature of the flare, defaults to 10_000*un.K
        :type T_flare: astropy.units.quantity.Quantity, optional
        :return: The energy in units of ergs
        :rtype: astropy.units.quantity.Quantity

        """
        E = 0
        for pt in flare_lc.flux:
            L = self.CalculateFlareLuminosity(pt,T_flare= T_flare)
            E = np.nansum((E + L * self.int_time).to('erg'))
        return E
    

    def CalculateSTD(self):
        """
        Calculates the standard deviation of the detrended, flagged light curve
        :return: The standard deviation of the light curve
        :rtype: float

        """
        if self.lc_arr == None:
            self.FlagLightCurves()
        std = []
        for i in range(len(self.lc_arr)):
            std.append(np.std(self.lc_flagged[i].flux[self.window:-self.window]/self.lc_median[i][self.window:-self.window]))
        self.std = np.nanmedian(std)
        return self.std
    

    
    def __MakeMiscInfoDict__(self):
        """
        Makes dictionary of information about the star and flare analysis and statistics
        :return: dictionary with the values for window, eclipse_window,flare_rate,median,std,tic_num,radius,temperature,and period
        :rtype: dictionary

        """
        d = {}
        d.update({'eclipse_window':int(self.eclipse_window)})
        d.update({'flare_rate':float(self.flare_rate.value)})
        d.update({'window':int(self.window)})
        d.update({'std':float(self.std)})
        d.update({'tic_num':int(self.star.tic_num)})
        d.update({'radius':float(self.star.radius)})
        d.update({'temperature':float(self.star.temperature)})
        d.update({'period':float(self.star.period)})
        d.update({'median':float(self.median)})
        d.update({'n_pts':float(self._n_pts)})
        return d
    
    def __MakeMetaDataDict__(self):
        """
        Makes dictionary with information from the light curve's metadata
        :return: DESCRIPTION
        :rtype: TYPE

        """
        d = dict(self.lc.meta)
        d.pop('PDC_VAR')
        d.pop('PDC_VARP')
        d.pop('PDC_EPT')
        d.pop('PDC_EPTP')
        d.pop('QUALITY_MASK')
        return d
    
    def WriteOutData(self,base_dir:str = None):
        """
        Writes out the class data to a folder in the base_dir
        :param base_dir: The name of the directory that the files get written to, defaults to None
        :type base_dir: str, optional

        """
        
        #Make the directory that the data gets written to:
        try:
            if base_dir is None:
                fp = 'TIC'+str(self.star.tic_num) + r'/'
            elif base_dir is not None:
                if base_dir[-1] != '/':
                    base_dir += r'/'
                fp =  base_dir + 'TIC'+str(self.star.tic_num)+ r'/'
            os.system('mkdir '+fp)
        except:
            raise Exception("Couldn't make directory "+fp)
            
        # Make dictionaries of the miscellaneous and lightcurve metadata and write them out:
        d = self.__MakeMiscInfoDict__()
        m = self.__MakeMetaDataDict__()
        f = open(fp+'star_information.json','w')
        json.dump(d,f)
        f.close()
        f = open(fp+'metadata.json','w')
        json.dump(m,f)
        f.close()
        
        # Write out data for the various lists of light curves:
        for i in range(len(self.lc_arr)):
            # Make various light curve file names:
            lc_arr_n = fp + 'raw_lc_section_'+"{:02d}".format(i)+'.npz'
            lc_flag_n = fp + 'flag_lc_section_'+"{:02d}".format(i)+'.npz'
            lc_norm_n = fp + 'norm_lc_section_'+"{:02d}".format(i)+'.npz'
            lc_median_n = fp + 'lc_median_section_'+"{:02d}".format(i)+'.npz'
            
            # Convert from LightCurve object to np.array() and write to npz:
            LightCurvetoNPZ(self.lc_arr[i],lc_arr_n)
            LightCurvetoNPZ(self.lc_flagged[i],lc_flag_n)
            LightCurvetoNPZ(self.lc_norm[i],lc_norm_n)
            
            #lc_median is already a list of arrays instead of LightCurve objects:
            np.savez(lc_median_n, self.lc_median[i],overwrite = True)
        
        # Write out the full light curve to an npz file:    
        lc_n = fp + 'full_lc.npz'
        LightCurvetoNPZ(self.lc,lc_n)
        
        # Write out the flare_table to an ascii file:
        flare_table_fn = fp + 'flare_table.tab'
        self.flare_table.write(flare_table_fn, format = 'ascii',overwrite = True)
        
        # Write out the data from the list of flares:
        for i in range(len(self.flares)):
            fn = fp + 'flare_'+"{:02d}".format(i)+'.npz'
            LightCurvetoNPZ(self.flares[i],fn)
        return
  

    
def PlanckFunction(lam, T):
    """
    The Planck function for calculating the specific flux for a given wavelength and temperature
    :param lam: wavelength in nm
    :type lam: float
    :param T: temperautre in K
    :type T: float
    :return: specific intensity in units W/m**2/nm
    :rtype: float

    """
    lam = lam*un.nm
    T = T*un.K
    h = const.h
    c = const.c
    k = const.k_B
    B_num = 2 * h*c**2/lam**5
    B_den =np.exp((h*c/(lam*k*T)).to('')) - 1
    B = (B_num/B_den).to('W/(m**2*nm)')
    return B.value

def NPZtoLightCurve(tab_n:str):
    """
    Converts an npz file to a LightCurve object
    :param tab_n: file name of the npz data
    :type tab_n: str
    :return: LightCurve object of the npz data
    :rtype: lk.LightCurve

    """
    # loads in data:
    arr = np.load(tab_n,allow_pickle = True)
    data = arr['data']
    
    # gets information on what values were masked:
    mask = arr['mask']
    col_names = arr['col_names']
    
    # assigns meta data to a dictionary:
    meta_dat = dict(arr['meta_dat'])
    
    # makes LightCurve object from the data and meta data:
    t = Table(data.transpose(),names=col_names)
    lc = lk.LightCurve(t,meta=meta_dat)
    
    # masks data in all columns:
    for c in col_names:
        try:
            mc = lc.MaskedColumn(lc[c], name = c,mask = mask)
            lc[c] = mc
        except:
            warn(f"Couldn't mask {c} column")
            pass

    return lc

def LightCurvetoNPZ(tab:lk.LightCurve,out_name:str):
    """
    Converts LightCurve object to npz file
    :param tab: the light curve object
    :type tab: lk.LightCurve
    :param out_name: base file name to write the data out to
    :type out_name: str

    """
    # makes array for all of the data of the light curve:
    col_names = tab.colnames
    arr = np.zeros((len(col_names),len(tab)))
    
    # gets mask information:
    mask = tab.flux.mask
    
    # assign light curve information to the appropriate array column
    for i in range(len(col_names)-1):
        arr[i,:] = tab[col_names[i]].value
        
    # write out data, mask information, column names, and meta data to file:
    np.savez(out_name,data = arr, mask = mask, col_names = col_names,meta_dat = list(tab.meta.items()))
    return

def LoadInStar(fp:str = ''):
    """
    Loads in information about the Star and Flares objects that were previously saved
    :param fp: file path of the base directory, defaults to ''
    :type fp: str, optional
    :return: Flares object with the loaded in data
    :rtype: Flares object

    """
    fp = os.path.join(fp, '')
    # Get file names for the various light curve lists:
    lc_arr_fns = glob.glob(fp + 'raw_lc_section_*.npz')
    lc_flag_fns = glob.glob(fp + 'flag_lc_section_*.npz')
    lc_norm_fns = glob.glob(fp + 'norm_lc_section_*.npz')
    lc_median_fns = glob.glob(fp + 'lc_median_section_*.npz')
    flare_fns = glob.glob(fp + 'flare*.npz')
    flare_fns.sort()
    lc_arr_fns.sort()
    lc_flag_fns.sort()
    lc_norm_fns.sort()
    lc_median_fns.sort()
    
    # Initialize lists for the various light curve lists:
    lc_arr = []
    lc_flag = []
    lc_norm = []
    lc_median = []
    flares = []
    
    # load in star information:
    misc = json.load(open(fp+'star_information.json'))
    
    # load in information for the various light curve lists:
    for i in range(len(lc_arr_fns)):
        lc_arr.append(NPZtoLightCurve(lc_arr_fns[i]))
        lc_flag.append(NPZtoLightCurve(lc_flag_fns[i]))
        lc_norm.append(NPZtoLightCurve(lc_norm_fns[i]))
        lc_median.append(np.load(lc_median_fns[i],allow_pickle=True)['arr_0'])
        
    # load in information for flares:    
    for i in range(len(flare_fns)):
        flares.append(NPZtoLightCurve(flare_fns[i]))
     
    # make light curve collection of lc_arr:    
    lcs = lk.LightCurveCollection(lc_arr)
    
    # make Star object from loaded in data:
    star = Star(misc['tic_num'], misc['radius'], misc['temperature'],period = misc['period'],lcs = lcs)
    
    # make Flares object from loaded in data:
    fl = Flares(star,process=False)
    
    # assign values to Flares object from misc data:
    fl.eclipse_window = misc['eclipse_window']
    fl.flare_rate = misc['flare_rate']/un.yr
    fl.window = misc['window']
    fl.median = misc['median']
    fl.std = misc['std']
    fl._n_pts = misc['n_pts']
    
    # read in flare_table:
    flare_table = Table.read(fp+'flare_table.tab',format = 'ascii')
    flags = []
    
    # convert flag information from flare_table from string to bool:
    for f in flare_table:
        if f['flag'] == 'True':
            flags.append(True)
        elif f['flag'] == 'False':
            flags.append(False)
    flare_table['flag'] = flags
    
    # assign rest of Flares object properties:
    fl.flare_table = flare_table
    fl.lc_arr = lc_arr
    fl.lc_flagged = lc_flag
    fl.lc_norm = lc_norm
    fl.lc_median = lc_median
    fl.lc = fl.star.lcs.stitch()
    fl.flares = flares
    return fl
