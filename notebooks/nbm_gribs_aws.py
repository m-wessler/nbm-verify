import os
import boto3

import numpy as np
import pandas as pd
import requests as req
import matplotlib.pyplot as plt

from botocore import UNSIGNED
from botocore.client import Config
from datetime import datetime, timedelta

from functools import partial
from multiprocessing import get_context, Pool, cpu_count
bb. 
tmp = '/scratch/general/lustre/u1070830/nbm/'
wgrib2 = '/uufs/chpc.utah.edu/sys/installdir/wgrib2/2.0.8/wgrib2/wgrib2'

nlon, xlon, nlat, xlat = -130, -100, 30, 50

os.environ['OMP_NUM_THREADS'] = '1'

def s3_list_files(bucket_name, prefix=''):
    
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = client.get_paginator("list_objects")

    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    keys = []
    for page in page_iterator:
        if "Contents" in page:
            for key in page["Contents"]:
                keyString = key["Key"]
                keys.append(keyString)

    return keys if keys else []

def s3_to_temp(object_name, bucket_name, file_name):
    
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    client.download_file(bucket_name, object_name, file_name)

def s3_to_temp_mp(obj):
    
    tmpdir = tmp + '%s/'%init.strftime('%Y%m%d')
    os.makedirs(tmpdir, exist_ok=True)
    
    tmpfile = tmpdir + os.path.basename(obj)
    subset = tmpfile.replace('.co.', '.WR.')

    if not os.path.isfile(subset):
    
        if not os.path.isfile(tmpfile):
            s3_to_temp(obj, bucket, tmpfile)
            print('%s saved'%os.path.basename(tmpfile))

        else:
            print('%s found, skipping'%os.path.basename(tmpfile))
            
        subset_grib(tmpfile, subset)
        
        os.remove(tmpfile)

    else:
        print('%s subset found, skipping'%os.path.basename(subset))

        
def subset_grib(full_file, subset_file):
        
    # Generate subset file using wgrib2
    print('\nSubsetting %s'%os.path.basename(subset_file))
    
    run_cmd = '%s %s -small_grib %d:%d %d:%d %s > ./tmp.txt'%(
        wgrib2, full_file, nlon, xlon, nlat, xlat, subset_file)
    
    os.system(run_cmd)
    
user = 'u1070830'

temp_dir = '/scratch/general/lustre/%s/nbm/'%user
os.makedirs(temp_dir, exist_ok=True)

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'noaa-nbm-grib2-pds'

start = datetime(2020, 10, 1, 0, 0)
end = datetime(2020, 10, 12, 23, 59)
freq = '6H'

date_range = pd.date_range(start, end, freq=freq)

for init in date_range:

    print(init)
    
    obj_list = s3_list_files(bucket, prefix='blend.%s/%02d/qmd/'%(init.strftime('%Y%m%d'), init.hour))
    obj_list = [f for f in obj_list if '.co.' in f]
    obj_list = [f for f in obj_list if '.idx' not in f]
    obj_list = [f for f in obj_list if int(f.split('.')[4].replace('f', ''))%6 == 0]
    obj_list = [f for f in obj_list if int(f.split('.')[4].replace('f', '')) >= 6]
    obj_list = [f for f in obj_list if int(f.split('.')[4].replace('f', '')) <= 180]

    workers = int(cpu_count()-1) if int(cpu_count()-1) <= len(obj_list) else len(obj_list)
    with get_context('fork').Pool(workers) as p:
        p.map(s3_to_temp_mp, obj_list, chunksize=1)
        p.close()
        p.join()