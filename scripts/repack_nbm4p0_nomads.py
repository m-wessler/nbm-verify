import os
import numpy as np
import xarray as xr
import netCDF4 as nc

from glob import glob
from functools import partial
from os import makedirs as mkdir
from multiprocessing import get_context
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREAD'] = '1'

# Set up init to use sys.argv later
init = datetime(2020, 10, 7, 12)

nlat, xlat = 30, 50
nlon, xlon = -130, -100

tmpdir = '/scratch/general/lustre/u1070830/nbm/tmp/'; mkdir(tmpdir, exist_ok=True)
datadir = '/scratch/general/lustre/u1070830/nbm/'; mkdir(datadir, exist_ok=True)

def download_grib(url, subset_str, tmp):
    from subprocess import Popen, call
    import requests
    
    filename = url.split('file=')[1].split('&')[0]
    filename = filename.replace('.co.', '.%s.'%subset_str)
        
    if not os.path.isfile(tmp + filename):
        print('Downloading %s'%filename)
        
        r = requests.get(url, allow_redirects=True)
        open(tmp + filename, 'wb').write(r.content)
        
        # cmd = 'wget -O "%s" "%s"'%(tmp + filename, url)
        # Popen(cmd, shell=True)
    
    return filename

def repack_nbm_grib2(f):
    import pygrib
    import gc
    
    print(f.split('/')[-1])
    
    if not os.path.isfile(f+'.nc'):

        try:
            grb = pygrib.open(f)
            msgs = grb.read()

            init = str(msgs[0]).split(':')[-2].split(' ')[-1]
            init = datetime.strptime(init, '%Y%m%d%H%M')

            fhr = msgs[0]['endStep']
            valid = np.datetime64(init + timedelta(hours=fhr))

            lons, lats = msgs[0].data()[2], msgs[0].data()[1]

        except:
            raise
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
                    percentile_labels.append([interval, msg['percentileValue']])

                else:
                    if got_deterministic[interval] == False:
                        deterministic_labels.append(interval)
                        deterministic.append(msg.values)
                        got_deterministic[interval] = True
                    else:
                        pass

            grb.close()
            gc.collect()

            deterministic_labels = np.array(deterministic_labels)
            deterministic_labels = deterministic_labels[np.argsort(deterministic_labels)]
            deterministic = np.array(deterministic)[np.argsort(deterministic_labels)]

            probability = np.array(probability, dtype=object).reshape(-1, lats.shape[0], lats.shape[1])
            probability_labels = np.array(probability_labels)

            percentile = np.array(percentile, dtype=object).reshape(-1, 99, lats.shape[0], lats.shape[1])
            percentile_labels = np.array(percentile_labels)

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
                            coords={'interval':('interval', np.unique(percentile_labels[:, 0])), 
                                    'percentile':('percentile', range(1, 100)),
                                    'lats':(('y', 'x'), lats), 'lons':(('y', 'x'), lons)})

            ds = xr.Dataset()

            # ds['fhr'] = fhr
            ds['time'] = valid
            ds.attrs['InitTime'] = str(init)

            ds['qpf'] = deterministic
            ds['pop'] = pop
            ds['probx'] = probability
            ds['pqpf'] = percentile

            ds.to_netcdf(f+'.nc')
            del ds
            gc.collect()

            return None

    else:
        print('Found: %s, skipping'%f.split('/')[-1])
        
def write_output(output, ncfilename):
    
    # ncfilename = './output.ncCustom.nc'
    lat, lon = output['lats'], output['lons']

    with nc.Dataset(tmpdir + ncfilename, 'w', format='NETCDF4') as ncfile:

        ncfile.nx = str(lon.shape[1])
        ncfile.ny = str(lon.shape[0])

        ncfile.InitTime = output.attrs['InitTime']

        # Lat Lon dimensions and data
        ncfile.createDimension('lon', lon.shape[1])
        ncfile.createDimension('lat', lon.shape[0])
        ncfile.createDimension('time', None)
        ncfile.createDimension('interval', output['interval'].size)
        ncfile.createDimension('percentile', output['percentile'].size)
        ncfile.createDimension('threshold', output['threshold'].size)

        lon_nc = ncfile.createVariable('lon', 'f4', ('lat', 'lon'))
        lon_nc.long_name = 'longitude'
        lon_nc.units = 'degrees_east'
        lon_nc.standard_name = 'longitude'
        lon_nc._CoordinateAxisType = 'Lon'

        lat_nc = ncfile.createVariable('lat', 'f4', ('lat', 'lon'))
        lat_nc.long_name = 'latitude'
        lat_nc.units = 'degrees_north'
        lat_nc.standard_name = 'latitude'
        lat_nc._CoordinateAxisType = 'Lat'

        lon_nc[:] = output.lons.values
        lat_nc[:] = output.lats.values

        interval = ncfile.createVariable('interval', 'short', ('interval'))
        interval.long_name = 'accumulation interval'
        interval.units = 'hours'
        interval.standard_name = 'interval'
        interval[:] = output['interval'].values.astype(int)

        percentile = ncfile.createVariable('percentile', 'short', ('percentile'), 
                                           zlib=True, complevel=9)
        percentile.long_name = 'accumulation percentile'
        percentile.units = 'none'
        percentile.standard_name = 'percentile'
        percentile[:] = output['percentile'].values.astype(int)

        threshold = ncfile.createVariable('threshold', 'f4', ('threshold'), 
                                          zlib=True, complevel=9)
        threshold.long_name = 'probabiity of exceedence threshold'
        threshold.units = 'in'
        threshold.standard_name = 'threshold'
        threshold[:] = output['threshold'].values

        # Write variable data
#         qpf_nc = ncfile.createVariable('qpf', 'f4', ('time', 'interval', 'lat', 'lon'), 
#                                        fill_value=-9999.0, zlib=True, complevel=9)
#         qpf_nc.long_name = 'Deterministic QPF'
#         qpf_nc.level = '0'
#         qpf_nc.units = 'in'
#         qpf_nc[:] = output['qpf'].values

#         pop_nc = ncfile.createVariable('pop', 'f4', ('time', 'interval', 'lat', 'lon'), 
#                                        fill_value=-9999.0, zlib=True, complevel=9)
#         pop_nc.long_name = 'Probability of Precipitation (> 0.01")'
#         pop_nc.level = '0'
#         pop_nc.units = 'in'
#         pop_nc[:] = output['pop'].values

        pqpf_nc = ncfile.createVariable('pqpf', 'f4', ('time', 'interval', 'percentile', 'lat', 'lon'), 
                                        fill_value=-9999.0, zlib=True, complevel=9)
        pqpf_nc.long_name = 'Probabilistic QPF'
        pqpf_nc.level = '0'
        pqpf_nc.units = 'in'
        pqpf_nc[:] = output['pqpf'].values

        probx_nc = ncfile.createVariable('probx', 'f4', ('time', 'interval', 'threshold', 'lat', 'lon'), 
                                         fill_value=-9999.0, zlib=True, complevel=9)
        probx_nc.long_name = 'Probability of Exceedence'
        probx_nc.level = '0'
        probx_nc.units = '%'
        probx_nc[:] = output['probx'].values

        print(ncfile)
    
if __name__ == '__main__':
    
    yyyy, mm, dd, hh = init.year, init.month, init.day, init.hour

    base = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl?'
    var = '&var_APCP=on'
    region = '&subregion=&leftlon={:.2f}&rightlon={:.2f}&toplat={:.2f}&bottomlat={:.2f}'.format(nlon, xlon, xlat, nlat)
    mdir = '&dir=%2Fblend.{:04d}{:02d}{:02d}%2F{:02d}%2Fqmd'.format(yyyy, mm, dd, hh)

    url_list = []

    # Need to fix the data processing below to allow for sub24 leads
    for fhr in np.arange(24, 180+1, 6):
        file = 'file=blend.t{:02d}z.qmd.f{:03d}.co.grib2'.format(hh, fhr)
        url_list.append(base + file + var + region + mdir)
        
    download_grib_mp = partial(download_grib, subset_str='WR', tmp=tmpdir)
    
    with get_context('forkserver').Pool(len(url_list)) as p:
        flist = p.imap_unordered(download_grib_mp, url_list, chunksize=1)
        p.close()
        p.join()

    flist = sorted(flist)
    filelist = sorted(glob(tmpdir + '*.grib2'))
        
    with get_context('forkserver').Pool(6) as p:
        output = p.imap_unordered(repack_nbm_grib2, filelist, chunksize=1)
        p.close()
        p.join()
    
    output = [xr.open_dataset(f+'.nc') for f in filelist]
    output = xr.concat([i for i in output if i is not None], dim='time')
    
    # print(output)
    
    write_output(output)

    # compress = {'compression':'gzip', 'compression_opts':9}
    # encoding = {var:compress for var in output.data_vars if var != 'time'}
    # output.to_netcdf(tmpdir + './test_output.nc', engine='h5netcdf', encoding=encoding)