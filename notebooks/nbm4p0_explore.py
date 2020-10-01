import os
import sys
import numpy as np
import xarray as xr

from glob import glob
from functools import partial
from os import makedirs as mkdir
from multiprocessing import get_context, cpu_count
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREADS'] = '1'

nlat, xlat = 47, 48 #33.5, 42.0
nlon, xlon = -122, -121 #-114.0, -104.5

mp_cores = cpu_count()*2
remove_gribs = False
mm_in = 1/25.4

def download_grib(url, subset_str, tmp):
    import requests
    
    # Needs filesize check and redownload attempts for junk
    
    filename = url.split('file=')[1].split('&')[0]
    filename = filename.replace('.co.', '.%s.'%subset_str)
    
    if not os.path.isfile(tmp + filename):
        print('Downloading: %s'%url)
        r = requests.get(url, allow_redirects=True)
        open(tmp + filename, 'wb').write(r.content)
    else:
        print('Exists, skipping: %s'%filename)
    
    return filename

def repack_nbm_grib2(f):
    import pygrib
    import gc
    
    try:
        grb = pygrib.open(f)
        msgs = grb.read()
                
        init = str(msgs[0]).split(':')[-2].split(' ')[-1]
        init = datetime.strptime(init, '%Y%m%d%H%M')
        
        fhr = msgs[0]['endStep']
        valid = np.array(init + timedelta(hours=fhr), dtype='datetime64[s]')
        
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
                    threshold = round(msg['upperLimit']*mm_in, 2)
                    probability.append([msg.values])
                    probability_labels.append([interval, threshold])

            elif 'percentileValue' in msg.keys():                
                percentile.append([msg.values])
                percentile_labels.append([interval, msg['percentileValue']])

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

        probability_labels = np.array(probability_labels)        
        probability = np.array(probability, dtype=object).reshape(
            -1, lats.shape[0], lats.shape[1])

        percentile_labels = np.array(percentile_labels)
        n_perc_intervals = np.unique(percentile_labels[:, 0]).size
        
        percentile = np.array(percentile, dtype=object).reshape(
            n_perc_intervals, 99, lats.shape[0], lats.shape[1])
        
        deterministic = xr.DataArray(deterministic.astype(np.float32)*mm_in, name='pop',
                        dims=('interval', 'x', 'y'),
                        coords={'interval':('interval', deterministic_labels),
                                'lats':(('x', 'y'), lats), 'lons':(('x', 'y'), lons)})

        pop = xr.DataArray(probability[:3].astype(np.float32), name='pop',
                        dims=('interval', 'x', 'y'),
                        coords={'interval':('interval', probability_labels[:3, 0]),
                                'lats':(('x', 'y'), lats), 'lons':(('x', 'y'), lons)})

        probability = xr.DataArray([probability[2:].astype(np.float32)], name='probability',
                        dims=('interval', 'threshold', 'x', 'y'),
                        coords={'interval':('interval', [24]), 'threshold':('threshold', probability_labels[2:,1]),
                                'lats':(('x', 'y'), lats), 'lons':(('x', 'y'), lons)})

        # print(fhr, percentile_labels, percentile.shape)
        percentile = xr.DataArray(percentile.astype(np.float32)*mm_in, name='percentile',
                        dims=('interval', 'percentile', 'x', 'y'),
                        coords={'interval':('interval', np.unique(percentile_labels[:, 0])), 
                                'percentile':('percentile', range(1, 100)),
                                'lats':(('x', 'y'), lats), 'lons':(('x', 'y'), lons)})

        ds = xr.Dataset()
        
        # ds['fhr'] = fhr
        ds['time'] = valid
        ds.attrs['InitTime'] = str(init)
        
        ds['qpf'] = deterministic
        ds['pop'] = pop
        ds['probx'] = probability
        ds['pqpf'] = percentile

        return ds
    
def write_netcdf(ds, ncfilename, complevel=9):
    import netCDF4 as nc
    from pandas import to_datetime as dt
    
    with nc.Dataset(ncfilename, 'w', format='NETCDF4') as ncfile:

        ncfile.nx = str(ds['lons'].shape[1])
        ncfile.ny = str(ds['lons'].shape[0])

        ncfile.InitTime = ds.attrs['InitTime']

        # Lat Lon dimensions and data
        ncfile.createDimension('lon', ds['lons'].shape[1])
        ncfile.createDimension('lat', ds['lons'].shape[0])
        ncfile.createDimension('time', ds['time'].size)
        ncfile.createDimension('interval', ds['interval'].size)
        ncfile.createDimension('percentile', ds['percentile'].size)
        ncfile.createDimension('threshold', ds['threshold'].size)

        lon_nc = ncfile.createVariable('lon', 'f4', ('lat', 'lon'), 
                                       zlib=True, complevel=complevel)
        lon_nc.long_name = 'Longitude'
        lon_nc.units = 'degrees_east'
        lon_nc.standard_name = 'longitude'
        lon_nc._CoordinateAxisType = 'Lon'

        lat_nc = ncfile.createVariable('lat', 'f4', ('lat', 'lon'), 
                                       zlib=True, complevel=complevel)
        lat_nc.long_name = 'Latitude'
        lat_nc.units = 'degrees_north'
        lat_nc.standard_name = 'latitude'
        lat_nc._CoordinateAxisType = 'Lat'

        lon_nc[:] = ds['lons'].values
        lat_nc[:] = ds['lats'].values

        time = ncfile.createVariable('time', 'f4', ('time'), 
                                     zlib=True, complevel=complevel)
        time.long_name = 'Valid Time'
        time.unit = "hours since 1970-01-01 00:00:00"
        time.standard_name = 'time'
        time[:] = ([(t.astype('datetime64[h]') - np.datetime64('1970-01-01T00:00:00')) / 
                    np.timedelta64(1, 'h') for t in output.time.values])
        
        interval = ncfile.createVariable('interval', 'short', ('interval'), 
                                         zlib=True, complevel=complevel)
        interval.long_name = 'Accumulation Interval'
        interval.units = 'hours'
        interval.standard_name = 'interval'
        interval[:] = ds['interval'].values.astype(int)

        percentile = ncfile.createVariable('percentile', 'short', ('percentile'), 
                                           zlib=True, complevel=complevel)
        percentile.long_name = 'Accumulation Percentile'
        percentile.units = 'none'
        percentile.standard_name = 'percentile'
        percentile[:] = ds['percentile'].values.astype(int)

        threshold = ncfile.createVariable('threshold', 'f4', ('threshold'), 
                                          zlib=True, complevel=complevel)
        threshold.long_name = 'Probabiity of Exceedence Threshold'
        threshold.units = 'in'
        threshold.standard_name = 'threshold'
        threshold[:] = ds['threshold'].values

        # Write variable data
        qpf_nc = ncfile.createVariable('qpf', 'f4', ('time', 'interval', 'lat', 'lon'), 
                                       fill_value=-9999.0, zlib=True, complevel=complevel)
        qpf_nc.long_name = 'Deterministic QPF'
        qpf_nc.level = '0'
        qpf_nc.units = 'in'
        qpf_nc[:] = ds['qpf'].values

        pop_nc = ncfile.createVariable('pop', 'f4', ('time', 'interval', 'lat', 'lon'), 
                                       fill_value=-9999.0, zlib=True, complevel=complevel)
        pop_nc.long_name = 'Probability of Precipitation (> 0.01")'
        pop_nc.level = '0'
        pop_nc.units = 'in'
        pop_nc[:] = ds['pop'].values

        pqpf_nc = ncfile.createVariable('pqpf', 'f4', ('time', 'interval', 'percentile', 'lat', 'lon'), 
                                        fill_value=-9999.0, zlib=True, complevel=complevel)
        pqpf_nc.long_name = 'Probabilistic QPF'
        pqpf_nc.level = '0'
        pqpf_nc.units = 'in'
        pqpf_nc[:] = ds['pqpf'].values

        probx_nc = ncfile.createVariable('probx', 'f4', ('time', 'interval', 'threshold', 'lat', 'lon'), 
                                         fill_value=-9999.0, zlib=True, complevel=complevel)
        probx_nc.long_name = 'Probability of Exceedence'
        probx_nc.level = '0'
        probx_nc.units = '%'
        probx_nc[:] = ds['probx'].values

    return None

if __name__ == '__main__':
    
    init = datetime.strptime(sys.argv[1], '%Y%m%d%H')
    yyyy, mm, dd, hh = init.year, init.month, init.day, init.hour

    base = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl?'
    var = '&var_APCP=on'
    region = '&subregion=&leftlon={:.2f}&rightlon={:.2f}&toplat={:.2f}&bottomlat={:.2f}'.format(nlon, xlon, xlat, nlat)
    mdir = '&dir=%2Fblend.{:04d}{:02d}{:02d}%2F{:02d}%2Fqmd'.format(yyyy, mm, dd, hh)

    url_list = []
    # Need to fix the data processing below to allow for sub24 leads
    for fhr in np.arange(6, 180+1, 6):
        file = 'file=blend.t{:02d}z.qmd.f{:03d}.co.grib2'.format(hh, fhr)
        url_list.append(base + file + var + region + mdir)

    tmpdir = '/Users/u1070830/Downloads/tmp/'; mkdir(tmpdir, exist_ok=True)
    datadir = '/Users/u1070830/Downloads/'; mkdir(datadir, exist_ok=True)

    download_grib_mp = partial(download_grib, subset_str='WR', tmp=tmpdir)

    with get_context('fork').Pool(mp_cores) as p:
        filelist_download = p.map(download_grib_mp, url_list, chunksize=1)
        p.close()
        p.join()

    filelist = sorted(glob(tmpdir + '*.grib2'))
    
    # Ensure all files were downloaded
    if ([f.split('/')[-1] for f in filelist] 
        == sorted([f for f in filelist_download])):

        with get_context('fork').Pool(mp_cores) as p:
            output = p.map(repack_nbm_grib2, filelist, chunksize=1)
            p.close()
            p.join()
            
        output = xr.concat([i for i in output if i is not None], dim='time')

        output_filename = 'blend.{}.t{:02d}z.qmd.WR.nc'.format(init.strftime('%Y%m%d'), hh)
        write_netcdf(output, ncfilename=tmpdir+output_filename)
        
        if remove_gribs:
            [os.remove(f) for f in filelist]
    
    else:
        print('Missing files, re-run %s'%init)
