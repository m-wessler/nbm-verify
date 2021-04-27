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

def extract_pqpf_verif_stats(_fhr, _urma, _idx, _mask):
    import gc
    
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

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)/100
        
        bins = np.arange(0, 101, 10)

        N = xr.where(~np.isnan(_nbm_masked_select), 1, 0).sum()
        n = xr.where(_urma_masked > thresh, 1, 0).sum()
        o = n/N
        uncertainty = o * (1 - o)
        
        reliability_inner = []
        resolution_inner = []
        reliability_diagram = []
        roc_diagram = []

        for i, bounds in enumerate(zip(bins[:-1], bins[1:])):

            left, right = np.array(bounds)/100
            center = round(np.mean([left, right]), 2)
            
            fk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), _nbm_masked_select, np.nan)
            nk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), 1, 0).sum()

            ok_count = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right) & (_urma_masked > thresh), 1, 0).sum()
            ok = ok_count/nk
            
            #        3D          1D     3D   1D
            _reliability_inner = nk * ((fk - ok)**2)
            _reliability_inner['center'] =left
            reliability_inner.append(_reliability_inner)

            #        1D         1D     1D   1D
            _resolution_inner = nk * ((ok - o)**2)
            _resolution_inner['center'] =left
            resolution_inner.append(_resolution_inner)
            
            reliability_diagram.append([center, ok.values])
                        
            hit = xr.where((_nbm_masked_select > left) & (_urma_masked > thresh), 1, 0).sum(dim='valid')
            false_alarm = xr.where((_nbm_masked_select > left) & (_urma_masked <= thresh), 1, 0).sum(dim='valid')
            
            observed_yes = xr.where(_urma_masked > thresh, 1, 0).sum(dim='valid')
            observed_no = xr.where(_urma_masked <= thresh, 1, 0).sum(dim='valid')
            forecasted_yes = xr.where(_nbm_masked_select > left, 1, 0).sum(dim='valid')
            
            hit_rate = hit/observed_yes
            false_alarm_rate = false_alarm/observed_no
            false_alarm_ratio = false_alarm/forecasted_yes
            freq_bias = forecasted_yes/observed_yes

            a_ref = (observed_yes * forecasted_yes) / nk
            miss = xr.where((_nbm_masked_select <= left) & (_urma_masked > thresh), 1, 0).sum(dim='valid')
            ets = (hit - a_ref) / (hit - a_ref + false_alarm + miss)
            
            roc_diagram.append([
                false_alarm_rate.mean().values, hit_rate.mean().values, 
                                left, 
                false_alarm_ratio.mean().values, ets.mean().values, 
                                freq_bias.mean().values])
        
        reliability_inner = xr.concat(reliability_inner, dim='center')
        reliability_inner = xr.where(_mask, reliability_inner.sum(dim='center'), np.nan)
        
        reliability = (1/N) * reliability_inner
        reliability = reliability.mean(dim='valid')
        
        resolution = (1/N) * xr.concat(resolution_inner, dim='center').sum(dim='center')
        
        brier = reliability - resolution + uncertainty
        brier_score = brier.mean().values
        
        brier_skill = 1 - (brier/o)
        brier_skill_score = brier_skill.mean().values
        
        brier = brier.rename('brier')
        brier_skill = brier_skill.rename('brier_skill')
        
        reliability_diagram = np.array(reliability_diagram).T
        roc_diagram = np.array(roc_diagram).T
        
        far = xr.DataArray(roc_diagram[0], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        hr = xr.DataArray(roc_diagram[1], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        
        faratio = xr.DataArray(roc_diagram[3], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        ets = xr.DataArray(roc_diagram[4], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        freq_bias = xr.DataArray(roc_diagram[5], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        
        data_merge = xr.merge([brier_skill.mean(dim=['x', 'y'])])

        # Need to figure out reliability scaling and add in here as (x, y)
        data_merge['n_events'] = observed_yes.sum(dim=['x', 'y'])
        data_merge['ets'] = ets
        data_merge['freq_bias'] = freq_bias
        data_merge['hit_rate'] = hr
        data_merge['false_alarm_rate'] = far
        data_merge['false_alarm_ratio'] = faratio

        data.append(data_merge)
        
        del fk, nk, ok_count, ok, reliability, reliability_inner, resolution
        del brier, brier_skill, brier_score, brier_skill_score, roc_diagram, reliability_diagram
        gc.collect()
    
    gc.collect()
                                        
    return xr.concat(data, dim='thresh')