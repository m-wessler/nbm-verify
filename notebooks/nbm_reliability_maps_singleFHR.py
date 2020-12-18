betterimport os, gc, sys
import pygrib
import regionmask
import cartopy
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt 

from glob import glob
from functools import partial
from matplotlib import gridspec
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'

# CONFIG # # CONFIG # # CONFIG # # CONFIG # # CONFIG # 
cwa = 'WESTUS'# 'SEW'#sys.argv[1]
fhr_start, fhr_end, fhr_step = 24, 24, 6

# start_date = datetime(2020, 5, 18, 0)
# end_date = datetime(2020, 10, 1, 0)

start_date = datetime(2020, 10, 1, 0)
end_date = datetime(2020, 12, 1, 0)

interval = 24
produce_thresholds = [0.01, 0.1, 0.25, 0.50, 1.0]
bint, bins_custom = 5, None

cwa_bounds = {
    'WESTUS':[30, 50, -130, -100],
    'SEW':[46.0, 49.0, -125.0, -120.5],
    'SLC':[37.0, 42.0, -114.0, -110],
    'MSO':[44.25, 49.0, -116.75, -112.25],
    'MTR':[35.75, 38.75, -123.5, -120.25],}

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
urma_dir = '/scratch/general/lustre/u1070830/urma/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
fig_dir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/nbm/'
os.makedirs(tmp_dir, exist_ok=True)
# CONFIG # # CONFIG # # CONFIG # # CONFIG # # CONFIG # 

def custom_cbar(_bins):
    if _bins[0] == 0:
        cmap = colors.ListedColormap(['#f5f5f5','#d8b365'])
        cbar_label = '\n                   Observed Relative Frequency        [Too Dry >]'
    elif _bins[1] == 100:
        cmap = colors.ListedColormap(['#5ab4ac','#f5f5f5'])
        cbar_label = '\n[< Too Wet]        Observed Relative Frequency                   '
    else:
        cmap = colors.ListedColormap(['#5ab4ac','#f5f5f5','#d8b365'])
        cbar_label = '\n[< Too Wet]        Observed Relative Frequency        [Too Dry >]'
    return [cmap, cbar_label]

def resize_colobar(event):
    # Tell matplotlib to re-draw everything, so that we can get
    # the correct location from get_position.
    plt.draw()

    posn = ax.get_position()
    colorbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                             0.04, axpos.height])

def extract_pbin_stats(_fhr, _urma):

    nbm_file = glob(nbm_dir + 'extract/nbm_probx_fhr%03d.nc'%_fhr)[0]
    
    # Subset the threshold value
    nbm = xr.open_dataset(nbm_file)['probx'].sel(
    y=slice(idx[0].min(), idx[0].max()),
    x=slice(idx[1].min(), idx[1].max()))

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
    nbm_mask, _nbm = xr.broadcast(mask, _nbm)
    urma_mask, _urma = xr.broadcast(mask, _urma)

    _nbm_masked = xr.where(nbm_mask, _nbm, np.nan)
    _urma_masked = xr.where(urma_mask, _urma, np.nan)
    
    #data = {}
    data = []
    
    for thresh in produce_thresholds:
        #data[thresh] = {}
        
        print('Processing f%03d %.2f"'%(_fhr, thresh))

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)

        for bins in zip(np.arange(0, 91, 10), np.arange(10, 101, 10)):

            b0, b1 = bins
            levels = np.unique([0, b0, b1, 100])

            # The meat and potatoes of the thing
            N = xr.where(
                    (_nbm_masked_select > b0) & 
                    (_nbm_masked_select <= b1), 
                1, 0).sum(dim='valid')

            n = xr.where(
                (_nbm_masked_select > b0) & 
                (_nbm_masked_select <= b1) & 
                (_urma_masked > thresh), 
                1, 0).sum(dim='valid')

            obs_rel_freq = xr.where(n > 5, n/N, np.nan)*100
            
            #data[thresh][_fhr] = [bins, n, N]
            data.append([thresh, _fhr, bins, n, N])
    
    return data

if __name__ == '__main__':

    extract_dir = nbm_dir + 'extract/'
    extract_flist = sorted(glob(extract_dir + '*'))

    if not os.path.isfile(urma_dir + 'agg/urma_agg.nc'):
        print('URMA aggregate not found')

    else:
        print('Getting URMA aggregate from file')
        urma = xr.open_dataset(urma_dir + 'agg/urma_agg.nc')['apcp24h_mm']

    urma = urma/25.4
    urma = urma.rename('apcp24h_in')
    lons, lats = urma.lon, urma.lat

    geodir = '../forecast-zones/'
    zones_shapefile = glob(geodir + '*.shp')[0]

    # Read the shapefile
    zones = gpd.read_file(zones_shapefile)

    # Prune to Western Region using TZ
    zones = zones.set_index('TIME_ZONE').loc[['M', 'Mm', 'm', 'MP', 'P']].reset_index()
    cwas = zones.dissolve(by='CWA').reset_index()[['CWA', 'geometry']]
    _cwas = cwas.copy()

    if cwa == 'WESTUS':
        _cwas['CWA'] = 'WESTUS'
        _cwas = _cwas.dissolve(by='CWA').reset_index()
        bounds = _cwas.total_bounds
    else:
        bounds = _cwas[_cwas['CWA'] == cwa].bounds.values[0]

    lons, lats = urma.lon, urma.lat
    mask = regionmask.mask_3D_geopandas(_cwas, lons, lats).rename({'region':'cwa'})
    mask['cwa'] = _cwas.iloc[mask.cwa]['CWA'].values.astype(str)
    mask = mask.sel(cwa=cwa)

    idx = np.where(
        (urma.lat >= bounds[1]) & (urma.lat <= bounds[3]) &
        (urma.lon >= bounds[0]) & (urma.lon <= bounds[2]))

    mask = mask.isel(y=slice(idx[0].min(), idx[0].max()), x=slice(idx[1].min(), idx[1].max()))
    urma = urma.isel(y=slice(idx[0].min(), idx[0].max()), x=slice(idx[1].min(), idx[1].max()))
    urma = urma.transpose('valid', 'y', 'x')

    fhrs = np.arange(fhr_start, fhr_end+1, fhr_step)
    extract_pbin_stats_mp = partial(extract_pbin_stats, _urma=urma)
    max_pool = 16 if cwa != 'WESTUS' else 4
    pool_count = len(fhrs) if len(fhrs) < max_pool else max_pool

    with mp.get_context('fork').Pool(pool_count) as p:
        returns = p.map(extract_pbin_stats_mp, fhrs)
        p.close()
        p.join()

    returns = np.array(returns, dtype=object).reshape(-1, 5)

    data = {fhr:{threshold:[] for threshold in produce_thresholds} for fhr in fhrs}

    for item in returns:
        threshold, fhr = item[:2]
        data[fhr][threshold].append(item[2:])
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  #    
    # BULK OF THE DATA PROCESSING DONE - PLOTTING ROUTINES BELOW #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # 
    
    