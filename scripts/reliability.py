import os, gc, sys
import pygrib
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import matplotlib.pyplot as plt 

from glob import glob
from functools import partial
from matplotlib import gridspec
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREADS'] = '1'

# CONFIG # # CONFIG # # CONFIG # # CONFIG # # CONFIG # 
cwa = sys.argv[1]
fhr_start, fhr_end, fhr_step = 24, 108, 6

start_date = datetime(2020, 10, 1, 0)
end_date = datetime(2020, 12, 3, 0)

produce_thresholds = [0.01, 0.25, 0.50]
bint, bins_custom = 5, None

cwa_bounds = {
    'WESTUS':[30, 50, -130, -110],
    'SEW':[46.0, 49.0, -125.0, -120.5],
    'SLC':[37.0, 42.0, -114.0, -110],
    'MSO':[44.25, 49.0, -116.75, -112.25],
    'MTR':[35.75, 38.75, -123.5, -120.25],}
# CONFIG # # CONFIG # # CONFIG # # CONFIG # # CONFIG # 

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
urma_dir = '/scratch/general/lustre/u1070830/urma/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
fig_dir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/nbm/'
os.makedirs(tmp_dir, exist_ok=True)

def calc_pbin(pbin, _bint, _thresh, _data, _urma):

    p0, p1 = pbin-_bint/2, pbin+_bint/2
    N = xr.where((_data >= p0) & (_data < p1), 1, 0).sum().values
    n = xr.where((_data >= p0) & (_data < p1) & (_urma > _thresh), 1, 0).sum().values
    
    return pbin, n, N

if __name__ == '__main__':
    
    extract_dir = nbm_dir + 'extract/'
    extract_flist = sorted(glob(extract_dir + '*'))
    
    if not os.path.isfile(urma_dir + 'agg/urma_agg.nc'):
        pass 
        #print('URMA aggregate not found')

    else:
        #print('Getting URMA aggregate from file')
        urma_whole = xr.open_dataset(urma_dir + 'agg/urma_agg.nc')['apcp24h_mm']

    urma_whole = urma_whole/25.4
    urma_whole = urma_whole.rename('apcp24h_in')

    pbin_stats_all = {}
    
    for thresh in produce_thresholds:
        
        for fhr in np.arange(fhr_start, fhr_end+1, fhr_step):

            open_file = [f for f in extract_flist if 'fhr%03d'%fhr in f][0]
            print(open_file)

            # Subset the times
            nbm = xr.open_dataset(open_file)
            nbm_time = nbm.valid
            urma_time = urma_whole.valid

            time_match = nbm_time[np.in1d(nbm_time, urma_time)].values

            time_match = np.array([t for t in time_match if pd.to_datetime(t) >= start_date])
            time_match = np.array([t for t in time_match if pd.to_datetime(t) <= end_date])

            nbm = nbm.sel(valid=time_match)
            urma = urma_whole.sel(valid=time_match)

            date0 = pd.to_datetime(time_match[0]).strftime('%Y/%m/%d %H UTC')
            date1 = pd.to_datetime(time_match[-1]).strftime('%Y/%m/%d %H UTC')
    
            nlat, xlat, nlon, xlon = cwa_bounds[cwa]

            lats, lons = nbm.lat, nbm.lon

            idx = np.where(
                (lats >= nlat) & (lats <= xlat) &
                (lons >= nlon) & (lons <= xlon))

            nbm = nbm.isel(x=slice(idx[1].min(), idx[1].max()), y=slice(idx[0].min(), idx[0].max()))
            urma = urma.isel(x=slice(idx[1].min(), idx[1].max()), y=slice(idx[0].min(), idx[0].max()))

            # Subset the threshold value
            nbm = nbm.sel(threshold=thresh)['probx']

            total_fc = xr.where(nbm > 0, 1, 0).sum()
            total_ob = xr.where(urma > thresh, 1, 0).sum()

            bins = np.arange(0, 101, bint)
            bins = bins_custom if bins_custom is not None else bins

            calc_pbin_mp = partial(calc_pbin, _bint=bint, _thresh=thresh,
                                   _data=nbm, _urma=urma)

            with mp.get_context('fork').Pool(len(bins)) as p:
                pbin_stats = p.map(calc_pbin_mp, bins, chunksize=1)
                p.close()
                p.join()

            pbin_stats_all[fhr] = np.array(pbin_stats, dtype=np.int)

        # Make the figure
        fig = plt.figure(figsize=(9, 11), facecolor='w') 
        axs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) 
        ax = plt.subplot(axs[0])
        ax1 = plt.subplot(axs[1])

        obs_freq_all, fcast_freq_all = [], []

        for fhr in pbin_stats_all.keys():
            pbin_stats = pbin_stats_all[fhr]

            # Calculate the reliability stats     
            fcast_prob = pbin_stats[:, 0]
            obs_freq = pbin_stats[:, 1]
            fcast_freq = pbin_stats[:, 2]
            obs_rel_freq = obs_freq/fcast_freq

            obs_freq_all.append(obs_freq)
            fcast_freq_all.append(fcast_freq)

            ax.plot(fcast_prob/100, obs_rel_freq, linewidth=1,
                    marker='+', markersize=5, label='F%03d'%fhr)

        obs_freq_all = np.array(obs_freq_all).sum(axis=0)
        fcast_freq_all = np.array(fcast_freq_all).sum(axis=0)
        obs_rel_freq_all = obs_freq_all/fcast_freq_all

        ax.plot(fcast_prob/100, obs_rel_freq_all, linewidth=3, color='r',
                marker='+', markersize=15, label='ALL')

        perfect = np.arange(0, 1.1, .1)
        climo = xr.where(urma > thresh, 1, 0).sum().values/urma.size
        skill = perfect - ((perfect - climo)/2)

        ax.plot(perfect, perfect, 
                color='k')

        ax.axhline(climo, 
                color='k', linestyle='--')

        ax.plot(perfect, skill, 
                color='k', linestyle='--')

        fillperf = np.arange(climo, 1, .001)
        ax.fill_between(fillperf, fillperf - (fillperf - climo)/2, 1,
                color='gray', alpha=0.35)

        fillperf = np.arange(0, climo, .001)
        ax.fill_between(fillperf, 0, fillperf - (fillperf - climo)/2,
                color='gray', alpha=0.35)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_xticks(perfect)
        ax.set_yticks(perfect)

        ax.set_xlabel('Forecast Probability')
        ax.set_ylabel('Observed Relative Frequency')
        ax.grid(zorder=1)

        ax.set_title((
            'NBM Reliability | CWA: %s\n'%cwa +
            '%s - %s\n'%(date0, date1) + 
            '%02dh Acc QPF | %3dh Lead Time\n\n'%(nbm.interval, nbm.fhr) +
            'Probability of Exceeding %.2f"\n\n'%thresh + 
            'n forecast prob > 0: %2.1e | n observed > %.2f: %2.1e'%(total_fc, thresh, total_ob)))

        ax.legend(loc='upper left')

        # # # # # # # # # # # # # # # # # # # # # # # #

        ax1.bar(bins, fcast_freq_all, color='k', width=4.5, zorder=10)

        ax1.set_xticks(bins[::2])
        ax1.set_xticklabels(bins[::2]/100)
        ax1.set_xlim([0, 100])

        ax1.set_yscale('log')
        ax1.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])

        ax1.set_xlabel('Forecast Probability')
        ax1.set_ylabel('# Forecasts')
        ax1.grid(zorder=-1)

        os.makedirs(fig_dir + 'reliability/%s/'%cwa, exist_ok=True)
        
        date0 = pd.to_datetime(time_match[0]).strftime('%Y%m%d%H')
        date1 = pd.to_datetime(time_match[-1]).strftime('%Y%m%d%H')
        save_meta = '%s_%s_%s_f%03d_f%03d_by%02d_probx%.2f_apcp%02dh.png'%(
            cwa, date0, date1, fhr_start, fhr_end, fhr_step, thresh, nbm.interval)
        
        savestr = fig_dir + 'reliability/%s/'%cwa + save_meta
        plt.savefig(savestr, dpi=100)
        plt.close()