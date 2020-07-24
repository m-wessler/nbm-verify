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

# Colorblind-friendly Palete
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
          "#0072B2", "#D55E00", "#CC79A7", "#999999"]

_site, _date0, _date1 = None, None, None
_datadir, _figdir = None, None
_thresholds, _thresh_id = None, None

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

def apcp_dist_plot(_obs, _nbm, _interval, show=False):
    
    iobs = _obs['%dh_precip_in'%_interval]
    iobs[iobs <= 0.01] = np.nan
    
    ifx = _nbm['APCP%dhr_surface'%_interval]
    ifx[ifx <= 0.01] = np.nan
    
    _threshold = np.append(0, np.nanpercentile(iobs, (33, 67, 100)))
    #_threshold = np.array([np.ceil(x*10)/10 for x in _threshold])
    
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), facecolor='w')
    ax, axx = axs
    
    maxval = max(np.nanmax(iobs), np.nanmax(iobs))
    binsize = 0.05
    
    ax.hist(iobs, bins=np.arange(0, maxval, binsize), 
            edgecolor='k', density=True, color=colors[0], alpha=0.75,
            label='Observed PDF (%.2f in bins)'%binsize)
    
    ax.hist(ifx, bins=np.arange(0, maxval, binsize), 
        edgecolor='k', density=True, color=colors[1], alpha=0.50,
        label='Forecast PDF (%.2f in bins)'%binsize)
    
    axx.hist(iobs, bins=np.arange(0, maxval, 0.00001), 
            density=True, cumulative=True, histtype='step', 
            linewidth=2.5, edgecolor=colors[0])
    axx.plot(0, linewidth=2.5, color=colors[0], label='Observed CDF (Continuous)')
    
    axx.hist(ifx, bins=np.arange(0, maxval, 0.00001), 
            density=True, cumulative=True, histtype='step', 
            linewidth=2.5, linestyle='-', edgecolor=colors[1])
    axx.plot(0, linewidth=2.5, linestyle='-', color=colors[1], label='Forecast CDF (Continuous)')
    
    for p, c in zip([33, 67], [colors[4], colors[5]]):
        ax.axvline(np.nanpercentile(iobs, p), color=c, linewidth=3, zorder=100, 
                   label='%dth Percentile _obs: %.2f in'%(p, np.nanpercentile(iobs, p)))
    
    axx.set_ylabel('\nCumulative [%]')
    axx.set_yticks([0, .2, .4, .6, .8, 1.0])
    axx.set_yticklabels([0, 20, 40, 60, 80, 100])
    axx.set_ylim([0, 1.01])
        
    for axi in axs:
        axi.set_xticks(np.arange(0, maxval+binsize, binsize*2))
        axi.set_xticklabels(['%.2f'%v for v in np.arange(0, maxval+binsize, binsize*2)], rotation=45)
        axi.set_xlim([0, maxval-binsize])
        axi.set_xlabel('\n%dh Forecast Precipitation [in]'%_interval)
        
        axi.set_ylabel('Frequency [%]\n')
        
        axi.set_title('%s\n%dh Forecast Precipitation\nNBM v3.2 Period %s – %s\n'%(
            _site, _interval, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d')))
        
        axi.grid(True)
        
    ax.legend(loc='upper right')
    axx.legend(loc='lower right')
  
    plt.tight_layout()
    
    savestr = '{}_{}h.APCP_dist.png'.format(_site, _interval)
    
    os.makedirs(_figdir + 'apcp_dist/', exist_ok=True)
    plt.savefig(_figdir + 'apcp_dist/' + savestr, dpi=150)
    
    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()
    
    return _threshold

def histograms_verif_rank(_data, _interval, _short, _long, _plot_type, _plot_var, _esize, show=False):
    
    select = _data[((_data['Interval'] == _interval)
                & ((_data['LeadTime'] >= _short) 
                & (_data['LeadTime'] <= _long)))]

    select = select[select['EventSize'] == _esize] if _esize != 'All' else select

    # Produce the actual reliability diagram
    font_size = 16
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(1, figsize=(10, 10), facecolor='w')

    pbinsize = 5
    pbins = np.arange(0, 101, pbinsize)

    hist = ax.hist(select[_plot_var], bins=np.append(-5, np.append(pbins, 105)), 
                   density=False, cumulative=False, alpha=0, zorder=-1)
    histy, histx = hist[0]/hist[0].sum()*100, hist[1][:-1]

    ax.bar(histx[1:-1]+2.5, histy[1:-1], width=5, color='0.75', edgecolor='k', zorder=5, label='QPF Rank %s'%_plot_type)
    ax.bar(histx[-1]+2.5, histy[-1], width=5, color='0.25', edgecolor='k', label='Outside P-Space', zorder=5)
    ax.bar(histx[0]+2.5, histy[0], width=5, color='0.25', edgecolor='k', zorder=5)

    ax.set_ylim([0, histy.max()+.5])

    try:
        ax.axvline(np.nanmean(select[_plot_var]), color=colors[0], linewidth=4, 
                   zorder=200, label='Mean: %d'%np.nanmean(select[_plot_var]))

        ax.axvline(np.nanpercentile(select[_plot_var], 50), color=colors[1], linewidth=4, 
                   zorder=200, label='Median: %d'%np.nanpercentile(select[_plot_var], 50))
    except:
        pass

    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlim([-5, 105])

    comp_type = 'Observation Verifies' if _plot_type == 'Verification' else 'Deterministic Compares'
    ax.set_xlabel('\n%s at Percentile'%(comp_type))

    ax.set_ylabel('Frequency (%)\n')

    n_precip_periods = np.unique(select['ValidTime'][~np.isnan(select['verif_ob'])]).shape[0]

    ax.set_title(('{} Percentile-Matched {}\nNBM v3.2 {} – {}\n\nEvent Size: {} ({:.2f} – {:.2f} in)\n' + 
                  'Interval: {} h | Lead Time: {} – {} h\nn={}, np={}\n').format(
                _site, _plot_type, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d'),
                _esize, _thresholds[_interval][_thresh_id[_esize][0]], _thresholds[_interval][_thresh_id[_esize][1]],
                _interval, _short, _long, len(select), n_precip_periods), size=font_size)

    ax.legend(loc='upper left', fontsize='small')
    ax.grid()
    plt.tight_layout()

    savestr = '{}_{}h_sz{}_lead{}-{}h.rankPDF.{}.png'.format(_site, _interval, _esize, _short, _long, _plot_type.lower())
    os.makedirs(_figdir + 'reliabilityCDF/', exist_ok=True)
    plt.savefig(_figdir + 'reliabilityCDF/' + savestr, dpi=150)

    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()