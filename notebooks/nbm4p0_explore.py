import os
import numpy as np
import xarray as xr

from glob import glob
from functools import partial
from os import makedirs as mkdir
from multiprocessing import get_context
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREADS'] = '1'

nlat, xlat = 40, 41
nlon, xlon = -112, -111

def download_grib(url, subset_str, tmp):
    import requests
    
    filename = url.split('file=')[1].split('&')[0]
    filename = filename.replace('.co.', '.%s.'%subset_str)
    
    print('Downloading: %s'%url)

    if not os.path.isfile(tmp + filename):
        r = requests.get(url, allow_redirects=True)
        open(tmp + filename, 'wb').write(r.content)
    
    return None

def repack_nbm_grib2(f):
    import pygrib
    import gc
    
    try:
        grb = pygrib.open(f)
        msgs = grb.read()
                
        init = str(msgs[0]).split(':')[-2].split(' ')[-1]
        init = datetime.strptime(init, '%Y%m%d%H%M')
        
        fhr = msgs[0]['endStep']
        valid = np.datetime64(init + timedelta(hours=fhr))
        
        lons, lats = msgs[0].data()[2], msgs[0].data()[1]
        
    except:
        return None
    
    else:
        probability, probability_labels = [], []
        percentile, percentile_labels = [], []

        deterministic, deterministic_labels = [], []
        got_deterministic = {i:False for i in [1, 6, 12, 24]}

        for msg in msgs:

            interval = msg['stepRange'].split('-')
            interval = int(interval[1]) - int(interval[0])

            if 'Probability of event' in str(msg):    
                    # Probability of event above upper limit (> 0.254) NOT inclusive
                    threshold = round(msg['upperLimit']/25.4, 2)
                    probability.append([msg.values])
                    probability_labels.append([interval, threshold])

            elif 'percentileValue' in msg.keys():                
                percentile.append([msg.values])
                # percentile_labels.append(str([interval, msg['percentileValue']]))

            else:
                if got_deterministic[interval] == False:
                    deterministic_labels.append(interval)
                    deterministic.append(msg.values)
                    got_deterministic[interval] = True
                else:
                    pass
                    # print('unused:', msg)

        grb.close()
        gc.collect()

        deterministic_labels = np.array(deterministic_labels)
        deterministic_labels = deterministic_labels[np.argsort(deterministic_labels)]
        deterministic = np.array(deterministic)[np.argsort(deterministic_labels)]

        probability = np.array(probability, dtype=object).reshape(-1, lats.shape[0], lats.shape[1])
        probability_labels = np.array(probability_labels)

        percentile = np.array(percentile, dtype=object).reshape(-1, 99, lats.shape[0], lats.shape[1])
        # percentile_labels = np.array(percentile_labels)

        deterministic = xr.DataArray(deterministic.astype(np.float32), name='pop',
                        dims=('interval', 'y', 'x'),
                        coords={'interval':('interval', deterministic_labels),
                                'lats':(('y', 'x'), lats), 'lons':(('y', 'x'), lons)})

        pop = xr.DataArray(probability[:3].astype(np.float32), name='pop',
                        dims=('interval', 'y', 'x'),
                        coords={'interval':('interval', probability_labels[:3, 0]),
                                'lats':(('y', 'x'), lats), 'lons':(('y', 'x'), lons)})

        probability = xr.DataArray([probability[2:].astype(np.float32)], name='probability',
                        dims=('interval', 'threshold', 'y', 'x'),
                        coords={'interval':('interval', [24]), 'threshold':('threshold', probability_labels[2:,1]),
                                'lats':(('y', 'x'), lats), 'lons':(('y', 'x'), lons)})

        percentile = xr.DataArray(percentile.astype(np.float32), name='percentile',
                        dims=('interval', 'percentile', 'y', 'x'),
                        coords={'interval':('interval', [6, 12, 24]), 'percentile':('percentile', range(1, 100)),
                                'lats':(('y', 'x'), lats), 'lons':(('y', 'x'), lons)})

        ds = xr.Dataset()
        
        # ds['fhr'] = fhr
        ds['time'] = valid
        ds.attrs['InitTime'] = str(init)
        
        ds['qpf'] = deterministic
        ds['pop'] = pop
        ds['probx'] = probability
        ds['pqpf'] = percentile

        return ds

if __name__ == '__main__':
    # Set up init to use sys.argv later
    init = datetime(2020, 9, 30, 12)
    yyyy, mm, dd, hh = init.year, init.month, init.day, init.hour

    base = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl?'
    var = '&var_APCP=on'
    region = '&subregion=&leftlon={:.2f}&rightlon={:.2f}&toplat={:.2f}&bottomlat={:.2f}'.format(nlon, xlon, xlat, nlat)
    mdir = '&dir=%2Fblend.{:04d}{:02d}{:02d}%2F{:02d}%2Fqmd'.format(yyyy, mm, dd, hh)

    url_list = []
    # Need to fix the data processing below to allow for sub24 leads
    for fhr in np.arange(24+6, 180+1, 6):
        file = 'file=blend.t{:02d}z.qmd.f{:03d}.co.grib2'.format(hh, fhr)
        url_list.append(base + file + var + region + mdir)

    tmpdir = '/Users/u1070830/Downloads/tmp/'; mkdir(tmpdir, exist_ok=True)
    datadir = '/Users/u1070830/Downloads/'; mkdir(datadir, exist_ok=True)

    download_grib_mp = partial(download_grib, subset_str='WR', tmp=tmpdir)

    with get_context('fork').Pool(16) as p:
        p.map(download_grib_mp, url_list, chunksize=1)
        p.close()
        p.join()

    filelist = sorted(glob(tmpdir + '*.grib2'))

    with get_context('fork').Pool(16) as p:
        output = p.map(repack_nbm_grib2, filelist, chunksize=1)
        p.close()
        p.join()    
        
    output = [repack_nbm_grib2(file) for file in filelist]
    output = xr.concat([i for i in output if i is not None], dim='time')

    compress = {'compression':'gzip', 'compression_opts':9}
    encoding = {var:compress for var in output.data_vars if var != 'time'}
    output.to_netcdf(tmp + './test_output.nc', engine='h5netcdf', encoding=encoding)
