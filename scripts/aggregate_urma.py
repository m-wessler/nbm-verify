import os, gc
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
n_cores = 64

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
urma_dir = '/scratch/general/lustre/u1070830/urma/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
os.makedirs(tmp_dir, exist_ok=True)

def open_urma(f, cfengine='pynio'):
    try:
        ds = xr.open_dataset(f, engine=cfengine)
        ds['valid'] = datetime.strptime(f.split('/')[-1].split('.')[1], '%Y%m%d%H')
    except:
        return None
    else:
        return ds

if __name__ == '__main__':
    
    urma_flist = sorted([f for f in glob(urma_dir + '*.WR.grib2') if 'idx' not in f])
    print(len(urma_flist), ' URMA files to read')

    print('Producing URMA aggregate')

    with mp.get_context('fork').Pool(n_cores) as p:
        urma = p.map(open_urma, urma_flist, chunksize=1)
        p.close()
        p.join()

    urma = [f for f in urma if f is not None]
    urma = xr.concat(urma, dim='valid').rename({'APCP_P8_L1_GLC0_acc':'apcp6h_mm', 
                                                'xgrid_0':'x', 'ygrid_0':'y',
                                                'gridlat_0':'lat', 'gridlon_0':'lon'})

    urma = urma['apcp6h_mm']
    
    urma24 = []
    for hr in [0, 6, 12, 18]:
        urma24.append(urma.resample(valid='24H', base=hr, closed='right', label='right').sum())
        
    urma24 = xr.concat(urma24, dim='valid').sortby('valid').rename('apcp24h_mm')

    nbm_times = [t for t in urma24.valid.values if pd.to_datetime(t) > datetime(2020, 5, 18, 0)]
    urma24 = urma24.sel(valid=nbm_times)

    os.makedirs(urma_dir + 'agg/', exist_ok=True)
    urma24.to_netcdf(urma_dir + 'agg/urma_agg.nc')