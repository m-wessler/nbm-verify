import os, sys
import numpy as np
from glob import glob
from multiprocessing import get_context

tmp = '/scratch/general/lustre/u1070830/urma/'

def subset_grib_spatial(full_file):
    
    subset_file = full_file.replace('grb2', 'WR.grib2')
        
    # Generate subset file using wgrib2
    nlon, xlon, nlat, xlat = -130, -100, 30, 50
    
    wgrib2 = '/uufs/chpc.utah.edu/sys/installdir/wgrib2/2.0.8/wgrib2/wgrib2'
    
    run_cmd = '%s %s -small_grib %d:%d %d:%d %s > ./tmp.txt'%(
        wgrib2, full_file, nlon, xlon, nlat, xlat, subset_file)
    
    #print(run_cmd)
    os.system(run_cmd)
    
if __name__ == '__main__':
    
    flist_subset = sorted(glob(tmp + 'urma*.WR.grib2'))
    date_subset = np.array([os.path.basename(f).split('.')[1] for f in flist_subset])

    flist = sorted(glob(tmp + 'urma*.grb2'))
    flist = [f for f in flist if os.path.basename(f).split('.')[1] not in date_subset]
    
    with get_context('fork').Pool(32) as p:
        p.map(subset_grib_spatial, flist, chunksize=1)
        p.close()
        p.join()