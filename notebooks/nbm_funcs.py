import os
import csv
import requests

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request as req
import scipy.stats as scipy

from datetime import datetime, timedelta

def get_1d_csv(get_req, this, total):
    
    _date, _init_hour, _url = get_req
    
    try:
        response = req.urlopen(_url).read().decode('utf-8')
        print('\r[%d/%d] %s %s'%(this, total, _date, _init_hour), end='')
        
    except:
        print('\r[%d/%d] NOT FOUND %s %s'%(this, total, _date, _init_hour), end='')
        return None
    
    else:
        init = datetime(_date.year, _date.month, _date.day, _init_hour, 0)

        response = response.split('\n')
        header = np.append('InitTime', response[0].split(','))
        
        lines = []
        for line in response[1:]:
            line = line.split(',')

            try:
                line[0] = datetime.strptime(line[0], '%Y%m%d%H')
            except:
                pass
            else:
                lines.append(np.append(init, line))
                        
        return header, lines
    

def get_precip_obs(s, d0, d1):

    # Tokens registered to michael.wessler@noaa.gov
    api_token = 'a2386b75ecbc4c2784db1270695dde73'
    api_key = 'Kyyki2tc1ETHUgShiscycW15e1XI02SzRXTYG28Dpg'
    base = 'https://api.synopticdata.com/v2/stations/precip?'
    
    allints = []
    
    forecast_interval = 6
    for interval in [6, 12, 24]:
        
        # Limit how big the observation lag can be (minutes)
        lag_limit = (interval/2)*60
        repeat = int((interval-forecast_interval)/6)+1
        
        df = []
        while repeat > 0:
            print('Working: Interval {}h Iteration {}'.format(interval, repeat))
                        
            _d0 = d0+timedelta(hours=(forecast_interval)*(repeat-1))
            _d1 = d1+timedelta(hours=1+forecast_interval*(repeat-1))
            
            url = base + 'stid={}&start={}&end={}&pmode=intervals&interval={}&token={}'.format(
                s,
                datetime.strftime(_d0, '%Y%m%d%H%M'),
                datetime.strftime(_d1, '%Y%m%d%H%M'),
                interval, api_token)
                        
            api_data_raw = requests.get(url).json()

            vdates = pd.date_range(_d0, _d1, freq='%dh'%interval)
            
            for i in api_data_raw['STATION'][0]['OBSERVATIONS']['precipitation']:
                
                if i['last_report'] is not None:
                    
                    try:
                        last_rep = datetime.strptime(i['last_report'], '%Y-%m-%dT%H:%M:%SZ')
                        vtime = vdates[np.argmin(np.abs(vdates - last_rep))]
                        lag_mins = (vtime - last_rep).seconds/60
                        value = float(i['total']) if lag_mins < lag_limit else np.nan
                    except:
                        #raise
                        pass
                    else:
                        #print('{}\t{}\t{}\t{}'.format(vtime, last_rep, lag_mins, value))
                        df.append([vtime, last_rep, lag_mins, value])
                    
            repeat -= 1

        allints.append(pd.DataFrame(df, 
            columns=['ValidTime', 'last_report', '%sh_lag_mins'%interval, '%sh_precip_mm'%interval]
            ).set_index('ValidTime').sort_index())

    return allints