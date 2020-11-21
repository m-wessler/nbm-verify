I can'timport os, gc
import pygrib
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp

from glob import glob
from functools import partial
from datetime import datetime, timedelta

os.environ['OMP_NUM_THREADS'] = '1'

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
urma_dir = '/scratch/general/lustre/u1070830/urma/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
os.makedirs(tmp_dir, exist_ok=True)

nbm_shape = (1051, 1132)

def unpack_fhr(nbm_file, xthreshold, xinterval, returned=False):
    
    # WE NEED TO MATCH URMA IN HERE IF WE CAN! 
        
    try:
        with pygrib.open(nbm_file) as grb:

            msgs = grb.read()
            if len(msgs) > 0:

                _init = nbm_file.split('/')[-2:]
                init = datetime.strptime(
                    _init[0] + _init[1].split('.')[1][1:-1], 
                    '%Y%m%d%H')

                if init.hour % 6 != 0:
                    init -= timedelta(hours=1)

                lats, lons = grb.message(1).latlons()

                valid = datetime.strptime(
                    str(msgs[0].validityDate) + '%02d'%msgs[0].validityTime, 
                    '%Y%m%d%H%M')

                step = valid - init
                lead = int(step.days*24 + step.seconds/3600)

                tmpfile = tmp_dir + '%02dprobX%s_%s_f%03d'%(xinterval, str(xthreshold).replace('.', 'p'), init.strftime('%Y%m%d%H'), lead)

                if not os.path.isfile(tmpfile + '.npy'):
                    print(nbm_file.split('/')[-2:])

                    for msg in msgs:

                        if 'Probability of event above upper limit' in str(msg):

                            interval = msg['stepRange'].split('-')
                            interval = int(interval[1]) - int(interval[0])

                            threshold = msg.upperLimit

                            if ((threshold == xthreshold)&(interval == xinterval)):

                                returned = True
                                agg_data = np.array((init, valid, lead, msg.values), dtype=object)
                                np.save(tmpfile, agg_data, allow_pickle=True)

                    if not returned:

                        agg_data = np.array((init, valid, lead, np.full(nbm_shape, fill_value=np.nan)), dtype=object)
                        np.save(tmpfile, agg_data, allow_pickle=True)

                else:
                    print(nbm_file.split('/')[-2:], 'from file')
                    agg_data = np.load(tmpfile + '.npy', allow_pickle=True)

            else:
                print('%s: No grib messages'%nbm_file.split('/')[-2:])
    
    except:
        pass
    
    else:
        
        try:
            grb.close()
            del grb
            gc.collect()	
            
        except:
            pass
        
        return agg_data
                
if __name__ == '__main__':
    
    # Pass data label to the extractor to pull out the variable we care about
    # Do these one at a time and save out the xarray to netcdf to compare w/ URMA
    extract_threshold = 0.254
    extract_interval = 24
    data_label = 'probx_%s_%02dh'%(str(extract_threshold).replace('.', 'p'), extract_interval)
    
    # Build a list of inits
    inits = pd.date_range(
        datetime(2020, 8, 15, 0), 
        datetime(2020, 10, 26, 23), 
        freq='6H')
    
    outfile = './' + data_label + '.%s_%s.WR.nc'%(
        inits[0].strftime('%Y%m%d%H'), 
        inits[-1].strftime('%Y%m%d%H'))
    
    if not os.path.isfile(outfile):
        
        nbm_flist_agg = []
        for init in inits:

            try:
                nbm_flist = sorted(glob(nbm_dir + init.strftime('%Y%m%d') + '/*t%02dz*'%init.hour))
                nbm_flist[0]

            except:
                nbm_flist = sorted(glob(nbm_dir + init.strftime('%Y%m%d') + '/*t%02dz*'%(init+timedelta(hours=1)).hour))

            nbm_flist = [f for f in nbm_flist if 'idx' not in f]

            if len(nbm_flist) > 0:
                nbm_flist_agg.append(nbm_flist)

        nbm_flist_agg = np.hstack(nbm_flist_agg)

        unpack_fhr_mp = partial(unpack_fhr, xinterval=extract_interval, xthreshold=extract_threshold)

        # 128 workers ~ 1.2GB RAM/worker
        workers = 86
        with mp.get_context('fork').Pool(workers) as p:
            returns = p.map(unpack_fhr_mp, nbm_flist_agg, chunksize=1)
            p.close()
            p.join()

        returns = np.array([r for r in returns if r is not None], dtype=object)
        init = returns[:, 0].astype(np.datetime64)
        valid = returns[:, 1].astype(np.datetime64).reshape(len(np.unique(init)), -1)
        lead = returns[:, 2].astype(np.int16).reshape(len(np.unique(init)), -1)
        data = np.array([r for r in returns[:, 3]], dtype=np.int8).reshape(len(np.unique(init)), -1, nbm_shape[0], nbm_shape[1])

        valid = xr.DataArray(valid, name='valid', dims=('init', 'lead'), coords={'init':np.unique(init), 'lead':np.unique(lead)})
        data = xr.DataArray(data, name=data_label, dims=('init', 'lead', 'y', 'x'), coords={'init':np.unique(init), 'lead':np.unique(lead)})
        data = xr.merge([data, valid])

        data.to_netcdf(outfile)
        
    else:
        data = xr.open_dataset(outfile)
    
    print('Done...')