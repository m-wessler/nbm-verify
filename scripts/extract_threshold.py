import os, sys, gc
import pygrib, cfgrib

import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from glob import glob
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREADS'] = '1'

upgrade_date = datetime(2020, 9, 29, 6)

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
os.makedirs(tmp_dir, exist_ok=True)

nlat, xlat = 30, 50
nlon, xlon = -130, -100

def unpack_fhr(nbm_file):
    
    if os.path.isfile(tmp_dir + os.path.basename(nbm_file) + '.nc'):
        return xr.open_dataset(tmp_dir + os.path.basename(nbm_file) + '.nc')
    
    else:
        if os.path.isfile(nbm_file):

            with pygrib.open(nbm_file) as grb:

                try:
                    lats, lons = grb.message(1).latlons()
                except:
                    data = None
                else:
                    idx = np.where(
                        (lats >= nlat) & (lats <= xlat) &
                        (lons >= nlon) & (lons <= xlon))

                    init_time = nbm_file.split('/')[-2:]
                    init_time = init_time[0] + init_time[1].split('.')[1][1:3]
                    init_time = datetime.strptime(init_time, '%Y%m%d%H')
                    valid_fhr = int(os.path.basename(nbm_file).split('/')[-1].split('.')[3][1:])

                    # Check if nbm3.2
                    if init_time.hour in [1, 7, 13, 19]:
                        init_time -= timedelta(hours=1)
                        valid_fhr += 1

                    valid_time = init_time + timedelta(hours=valid_fhr)
                    #print(init_time, valid_fhr, valid_time)
                    print(valid_fhr, valid_time)

                    percentile, probability, deterministic = [], [], []
                    percentile_labels, probability_labels, deterministic_labels = [], [], []

                    data = []
                    for msg in grb.read():

                        interval = msg['stepRange'].split('-')
                        interval = int(interval[1]) - int(interval[0])

                        if interval == 24:

                            if 'Probability of event' in str(msg):

                                threshold = round(msg['upperLimit']/25.4, 2)

                                if threshold in [0.01, 0.10, 0.25, 0.50, 1.00, 2.00]:
                                    
                                    idata = xr.DataArray(msg.data()[0].astype(np.float32), name='probx',
                                                         dims=('y', 'x'), 
                                                         coords={'lat':(('y', 'x'), lats), 
                                                                 'lon':(('y', 'x'), lons)})
                                    
                                    idata['init'] = init_time
                                    idata['valid'] = valid_time
                                    idata['fhr'] = valid_fhr
                                    idata['interval'] = interval
                                    idata['threshold'] = threshold

                                    data.append(idata)
                                                                    
            gc.collect()

            try:
                data = xr.concat(data, dim='threshold')
                
            except:
                return None
            
            else:
                data_slice = data.isel(x=slice(idx[1].min(), idx[1].max()), 
                      y=slice(idx[0].min(), idx[0].max()))
                
                # data_slice.to_netcdf(tmp_dir + os.path.basename(nbm_file) + '.nc')
                return data_slice

        else:
            return None
    
if __name__ == '__main__':

    init_hour = 0
    init_freq = '6H'
    
    # TEST THE NEW CODE ON A SMALLER SUBSET!!!!
    inits = pd.date_range(
        datetime(2020, 5, 15, init_hour, 0),
        datetime(2020, 12, 15, init_hour, 0),
        freq=init_freq)
        
    for forecast_hour in np.arange(24, 180+1, 6):

        outdir = nbm_dir + 'extract/'
        os.makedirs(outdir, exist_ok=True)
        outfile = 'nbm_probx_fhr%03d.nc'%forecast_hour

        if not os.path.isfile(outdir+outfile):

            flist = []
            for init in inits:

                if init < upgrade_date:
                    init += timedelta(hours=1)
                    _forecast_hour = forecast_hour-1
                else:
                    _forecast_hour = forecast_hour

                search_str = nbm_dir + '%s/*t%02dz*f%03d*WR.grib2'%(
                    init.strftime('%Y%m%d'), init.hour, _forecast_hour)
                search = glob(search_str)

                if len(search) > 0:
                    flist.append(search[0])

            flist = np.array(sorted(flist))
            print('nfiles: ', len(flist))
            
            with mp.get_context('fork').Pool(64) as p:
                returns = p.map(unpack_fhr, flist, chunksize=1)
                p.close()
                p.join()

            returns = [item for item in returns if item is not None]
            returns = xr.concat(returns, dim='valid')

            returns.to_netcdf(outdir + outfile)
            print('Saved %s'%(outdir + outfile))

            del returns
            gc.collect()