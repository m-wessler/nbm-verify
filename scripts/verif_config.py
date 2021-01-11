from datetime import datetime

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
urma_dir = '/scratch/general/lustre/u1070830/urma/'
tmp_dir = '/scratch/general/lustre/u1070830/tmp/'
fig_dir = '/uufs/chpc.utah.edu/common/home/steenburgh-group10/mewessler/nbm/'

# Start date and end date for stats
# First data (2020, 5, 18, 0)
# NBM v3.2 > v4.0 @ (2020, 9, 30, 12)

ver = '3p2'
start_date = datetime(2020, 6, 1, 0)
end_date = datetime(2020, 10, 1, 0)

# ver = '4p0'
# start_date = datetime(2020, 10, 1, 0)
# end_date = datetime(2021, 1, 1, 0)

# First, last forecast hour and interval to use
fhr_start, fhr_end, fhr_step = 24, 168, 24
interval = 24

# Produce stats for these predefined NBM PQPF Thresholds
produce_thresholds = [0.01, 0.1, 0.25, 0.50, 1.0]

# Provide bin interval or custom bins
bint = 10
bins_custom = None

# Minimum number of events required for spatial reliability result
n_events = 3

# Condense the lat lon grid with box filter
# Right now this only applies to the reliability maps
# Need to experiment with generalizing this
cx = 4 #8 if cwa == 'WESTUS' else 4
cy = cx
