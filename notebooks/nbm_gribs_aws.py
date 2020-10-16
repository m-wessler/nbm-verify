import os
import gzip
import pygrib
import boto3
import requests as req
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from botocore import UNSIGNED
from botocore.client import Config
from datetime import datetime, timedelta

from functools import partial
from multiprocessing import get_context, Pool, cpu_count

os.environ['OMP_NUM_THREADS'] = '1'

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
tmp = '/scratch/general/lustre/u1070830/nbm/'

def s3_list_files(bucket_name, prefix=''):
    
    paginator = client.get_paginator('list_objects')

    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    keys = []
    for page in page_iterator:
        if 'Contents' in page:
            for key in page['Contents']:
                keyString = key['Key']
                keys.append(keyString)

    return keys if keys else []

def s3_to_temp(obj, compress=False):
    
    obj_init = obj.split('/')[0].split('.')[-1]
    
    tmpdir = tmp + '%s/'%obj_init
    os.makedirs(tmpdir, exist_ok=True)
    
    tmpfile = tmpdir + os.path.basename(obj)
    subset = tmpfile.replace('.co.', '.WR.').replace('.master.', '.qmd.')
    subset_gz = subset + '.gz'
    
    print('Processing: %s'%tmpfile)

    if ((compress == True) & os.path.isfile(subset) & (not os.path.isfile(subset_gz))):
        compress_grib(subset)
        
    elif ((compress == False) & (os.path.isfile(subset))):
        pass
    
    else:
        
        if not os.path.isfile(tmpfile):
            client.download_file(bucket, obj, tmpfile)

        else:
            pass
            
        subset_grib_spatial(tmpfile, subset)
        os.remove(tmpfile)
        
        subset_grib_variables(subset, compress=compress)
        
    return subset_gz
        
def subset_grib_spatial(full_file, subset_file):
        
    # Generate subset file using wgrib2
    nlon, xlon, nlat, xlat = -130, -100, 30, 50
    
    wgrib2 = '/uufs/chpc.utah.edu/sys/installdir/wgrib2/2.0.8/wgrib2/wgrib2'
    
    run_cmd = '%s %s -small_grib %d:%d %d:%d %s > ./tmp.txt'%(
        wgrib2, full_file, nlon, xlon, nlat, xlat, subset_file)
    
    os.system(run_cmd)
    
def subset_grib_variables(full_file, compress):
    
    if compress:
        subset_file = full_file.replace('.master.', '.qmd.') + '.gz'
        with pygrib.open(full_file) as grib, gzip.open(subset_file, 'wb') as grib_out:
            
            # Isolate the precip strings, excluding less than 6h precip
            msgs = (s for s in grib.read() 
                    if (('Precipitation' in str(s)) & 
                        ((s.endStep - s.startStep) >= 6)))
            
            for msg in msgs:
                grib_out.write(msg.tostring())
        os.remove(full_file)
                
    else:
        subset_file = full_file.replace('.master.', '.qmd.') + '.tmp'
        with pygrib.open(full_file) as grib, open(subset_file, 'wb') as grib_out:
            
            # Isolate the precip strings, excluding less than 6h precip
            msgs = (s for s in grib.read() 
                    if (('Precipitation' in str(s)) & 
                        ((s.endStep - s.startStep) >= 6)))
            
            for msg in msgs:
                grib_out.write(msg.tostring())
                
        os.remove(full_file)
        os.rename(subset_file, subset_file.replace('.tmp', ''))
        
def compress_grib(full_file):
   
    subset_file = full_file.replace('.master.', '.qmd.') + '.gz'
    with pygrib.open(full_file) as grib, gzip.open(subset_file, 'wb') as grib_out:
        msgs = grib.read()
        for msg in msgs:
            grib_out.write(msg.tostring())
            
    os.remove(full_file)

bucket = 'noaa-nbm-grib2-pds'
os.makedirs(tmp, exist_ok=True)

# Determine the start of the qmd files and append two pd.date_range() sets
start_3p2 = datetime(2020, 5, 18, 1, 0)
end_3p2 = datetime(2020, 9, 29, 1, 0)
date_range_3p2 = pd.date_range(start_3p2, end_3p2, freq='6H')

start_4p0 = datetime(2020, 9, 29, 6, 0)
end_4p0 = datetime.now() - timedelta(hours=3)
date_range_4p0 = pd.date_range(start_4p0, end_4p0, freq='6H')

date_range = pd.to_datetime(np.append(date_range_3p2, date_range_4p0))
date_range = date_range[:28]

obj_list_stacked = []
for init in date_range:

    print('\rBuilding file list: %s'%init, end='')
    
    try:
        # NBM 4.0, QMD isolated
        obj_list = s3_list_files(bucket,
                                     prefix='blend.%s/%02d/qmd/'%(
                                         init.strftime('%Y%m%d'), init.hour))
        obj_list[0]
    
    except:
        # NBM 3.2, QMD inline, prior to 9/29/2020
        obj_list = s3_list_files(bucket,
                                     prefix='blend.%s/%02d/grib2/'%(
                                         init.strftime('%Y%m%d'), init.hour)) ####
        # NBM 3.2 PQPF on 1, 7, 13, 19
        obj_list = [f for f in obj_list if '.master.' in f]
        obj_list = [f for f in obj_list if (int(f.split('.')[4].replace('f', ''))+1)%6 == 0]
        
    else:
        # NBM 4.0 PQPF on 0, 6, 12, 18
        obj_list = [f for f in obj_list 
                        if int(f.split('.')[4].replace('f', ''))%6 == 0]

    # Isolate CONUS, ditch .idx, keep hours 6-180 (modifiable)
    obj_list = [obj for obj in obj_list if (
        ('.co.' in obj) & ('.idx' not in obj) &
        (5 <= int(obj.split('.')[4].replace('f', ''))) &
        (int(obj.split('.')[4].replace('f', '')) <= 180))]
    
    obj_list_stacked.append(obj_list)
    
obj_list_stacked = np.hstack(obj_list_stacked)

cpus = cpu_count()-1
nfiles = len(obj_list_stacked)
workers = cpus if cpus < nfiles else nfiles

with get_context('fork').Pool(workers) as p:

    s3_to_temp_mp = partial(s3_to_temp, compress=False)
    file_list = p.map(s3_to_temp_mp, obj_list_stacked, chunksize=1)
    p.close()
    p.join()
