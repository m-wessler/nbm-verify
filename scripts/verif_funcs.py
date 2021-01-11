import numpy as np
import xarray as xr
import pandas as pd

from glob import glob
from matplotlib import colors

from verif_config import *

def custom_cbar(_bins):
    
    base_colors = ['#a6611a','#dfc27d','white','#80cdc1','#018571'][::-1]
    lev = np.unique([0, _bins[0]-10, _bins[0], _bins[1]+10, _bins[1], 100])
    lev = lev[((lev >= 0) & (lev <= 100))]

    if _bins[0] == 0:
        cmap = colors.ListedColormap(base_colors[-3:])
        cbar_label = '\n                   Observed Relative Frequency [1 Bin Too Dry >] [Too Dry >]'
    
    elif _bins[1] == 100:
        cmap = colors.ListedColormap(base_colors[:3])
        cbar_label = '\n[< Too Wet] [< 1 Bin Too Wet] Observed Relative Frequency                   '
    
    else:
        cbar_label = ('\n[< Too Wet]  [< 1 Bin Too Wet]                [1 Bin Too Dry >]  [Too Dry >]' + 
                     '\n\nObserved Relative Frequency')
        
        if _bins[0] == 10:
            cmap = colors.ListedColormap(base_colors[1:5])
        elif _bins[1] == 90:
            cmap = colors.ListedColormap(base_colors[-5:-1])
        else:
            cmap = colors.ListedColormap(base_colors)
    
    cbar_label = '\nObserved Relative Frequency\nGray where n observed events < 5'
    
    return [lev, cmap, cbar_label]

def extract_pbin_stats(_fhr, _urma, _idx, _mask):

    nbm_file = glob(nbm_dir + 'extract/nbm_probx_fhr%03d.nc'%_fhr)[0]
    
    # Subset the threshold value
    nbm = xr.open_dataset(nbm_file)['probx'].sel(
    y=slice(_idx[0].min(), _idx[0].max()),
    x=slice(_idx[1].min(), _idx[1].max()))

    # Subset the times
    nbm_time = nbm.valid
    urma_time = _urma.valid
    time_match = nbm_time[np.in1d(nbm_time, urma_time)].values
    time_match = np.array([t for t in time_match if pd.to_datetime(t) >= start_date])
    time_match = np.array([t for t in time_match if pd.to_datetime(t) <= end_date])
    date0 = pd.to_datetime(time_match[0]).strftime('%Y/%m/%d %H UTC')
    date1 = pd.to_datetime(time_match[-1]).strftime('%Y/%m/%d %H UTC')

    _nbm = nbm.sel(valid=time_match)
    _urma = _urma.sel(valid=time_match)
    nbm_mask, _nbm = xr.broadcast(_mask, _nbm)
    urma_mask, _urma = xr.broadcast(_mask, _urma)

    _nbm_masked = xr.where(nbm_mask, _nbm, np.nan)
    _urma_masked = xr.where(urma_mask, _urma, np.nan)
    
    data = []
    
    for thresh in produce_thresholds:
        
        print('Processing f%03d %.2f"'%(_fhr, thresh))

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)

        for bins in zip(np.arange(0, 101-bint, bint), np.arange(0+bint, 101, bint)):

            b0, b1 = bins
            center = b1-bint
            levels = np.unique([0, b0, b1, 100])

            N = xr.where(
                    (_nbm_masked_select > b0) & 
                    (_nbm_masked_select <= b1), 
                1, 0).sum(dim='valid')

            n = xr.where(
                (_nbm_masked_select > b0) & 
                (_nbm_masked_select <= b1) & 
                (_urma_masked > thresh), 
                1, 0).sum(dim='valid')
                        
            # hit, yesFx/yesOb
            a = xr.where(
                (_nbm_masked_select >= center) &
                (_urma_masked > thresh),
                1, 0).sum(dim='valid')
            
            # false alarm, yesFx/noOb
            b = xr.where(
                (_nbm_masked_select >= center) &
                ((_urma_masked <= thresh)),#|np.isnan(_urma_masked)),
                1, 0).sum(dim='valid')
            
            # miss, noFx/yesOb
            c = xr.where(
                (_nbm_masked_select <= center) &
                (_urma_masked > thresh),
                1, 0).sum(dim='valid')
            
            # correct negative, noFx/noOb
            d = xr.where(
                (_nbm_masked_select <= center) &
                (_urma_masked <= thresh),
                1, 0).sum(dim='valid')
            
            # # # # # #
            
            obs_rel_freq = xr.where(n > 5, n/N, np.nan)*100
            
            data.append([thresh, _fhr, bins, n, N, [a, b, c, d]])
    
    return data

def extract_brier(_fhr, _urma, _idx, _mask):

    nbm_file = glob(nbm_dir + 'extract/nbm_probx_fhr%03d.nc'%_fhr)[0]
    print(nbm_file)
    
    # Subset the threshold value
    nbm = xr.open_dataset(nbm_file)['probx'].sel(
    y=slice(_idx[0].min(), _idx[0].max()),
    x=slice(_idx[1].min(), _idx[1].max()))

    # Subset the times
    nbm_time = nbm.valid
    urma_time = _urma.valid
    time_match = nbm_time[np.in1d(nbm_time, urma_time)].values
    time_match = np.array([t for t in time_match if pd.to_datetime(t) >= start_date])
    time_match = np.array([t for t in time_match if pd.to_datetime(t) <= end_date])
    date0 = pd.to_datetime(time_match[0]).strftime('%Y/%m/%d %H UTC')
    date1 = pd.to_datetime(time_match[-1]).strftime('%Y/%m/%d %H UTC')

    _nbm = nbm.sel(valid=time_match)
    _urma = _urma.sel(valid=time_match)
    nbm_mask, _nbm = xr.broadcast(_mask, _nbm)
    urma_mask, _urma = xr.broadcast(_mask, _urma)

    _nbm_masked = xr.where(nbm_mask, _nbm, np.nan)
    _urma_masked = xr.where(urma_mask, _urma, np.nan)
        
    data = []
    
    for thresh in produce_thresholds:
        
        print('Processing f%03d %.2f"'%(_fhr, thresh))

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)

        Y = _nbm_masked_select/100
        O = xr.where(_urma_masked > thresh, 1, 0)
        
        # Use observed climo or model climo ???
        P = O.sum(dim='valid')/O.valid.size
        P, _ = xr.broadcast(P, O)
        
        # Not sure why we're getting issues with np.inf, but correct for these here
        BS = ((Y - O)**2).mean(dim='valid')
        BS = xr.where(~np.isinf(BS), BS, np.nan)
        BS.name = 'BS'
        
        BS_cl = ((P - O)**2).mean(dim='valid')
        BS_cl = xr.where(~np.isinf(BS_cl), BS_cl, np.nan)
        BS_cl.name = 'BS_cl'
        
        BSS = 1 - (BS/BS_cl)
        BSS = xr.where(~np.isinf(BSS), BSS, np.nan)
        BSS.name = 'BSS'
        
        data.append(xr.merge([BS, BS_cl, BSS]))

    return xr.concat(data, dim='thresh')