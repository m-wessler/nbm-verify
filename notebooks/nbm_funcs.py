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
_init_hours, _dates = None, None
_datadir, _figdir = None, None
_thresholds, _thresh_id = None, None

def get_1d_csv(get_req, this, total, verbose=True):
    
    _date, _init_hour, _url = get_req
    
    try:
        response = req.urlopen(_url).read().decode('utf-8')
        if verbose:
            print('\r[%d/%d] %s %s'%(this, total, _date, _init_hour), end='')
        
    except:
        if verbose:
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
    

def get_precip_obs(s, d0, d1, verbose=False):

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
            if verbose:
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
                _esize, _thresholds[_interval][_thresh_id[_esize][0]], 
                _thresholds[_interval][_thresh_id[_esize][1]],
                _interval, _short, _long, len(select), n_precip_periods), size=font_size)

    ax.legend(loc='upper left', fontsize='small')
    ax.grid()
    plt.tight_layout()

    savestr = '{}_{}h_sz{}_lead{}-{}h.rankPDF.{}.png'.format(
        _site, _interval, _esize, _short, _long, _plot_type.lower())
    os.makedirs(_figdir + 'reliabilityCDF/', exist_ok=True)
    plt.savefig(_figdir + 'reliabilityCDF/' + savestr, dpi=150)

    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()
        
def reliability_verif_cdf(_data, _interval, _short, _long, _plot_type, _plot_var, _esize, show=False):
    
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

    hist = ax.hist(select[_plot_var], bins=pbins, density=True, cumulative=True,
            color='w', alpha=0, linewidth=3.5, zorder=10)
    histy, histx = hist[0]*100, hist[1][1:]-(pbinsize/2)

    ax.plot(histx, histy, marker='x', linestyle='--', markersize=10, color='k', linewidth=2)

    ax.plot(np.arange(0, 101, 1), np.arange(0, 101, 1), '--k', linewidth=1, zorder=20)

    ax.set_xticks(np.arange(0, 101, 5))
    ax.set_xlim([0, 101])

    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_yticklabels(np.arange(0, 101, 5), rotation=30)
    ax.set_ylim([0, 101])

    ax.set_yticklabels(np.arange(100, -5, -5), rotation=30)

    ax.set_ylabel('\nObserved in % of Forecasts')
    ax.set_xlabel('\nForecast Verifies At/Above Percentile\n')

    n_precip_periods = np.unique(select['ValidTime'][~np.isnan(select['verif_ob'])]).shape[0]
    ax.set_title(('{} Percentile-Matched {}\nNBM v3.2 {} – {}\n\nEvent Size: {} ({:.2f} – {:.2f} in)\n' + 
                  'Interval: {} h | Lead Time: {} – {} h\nn={}, np={}\n').format(
                _site, _plot_type, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d'),
                _esize, _thresholds[_interval][_thresh_id[_esize][0]], 
                _thresholds[_interval][_thresh_id[_esize][1]],
                _interval, _short, _long, len(select), n_precip_periods), size=font_size)

    ax.text(5, 92, 'Wet Bias')
    ax.text(85, 6, 'Dry Bias')
    ax.text(35, 38, 'Unbiased Distribution', rotation=40)

    # ax.legend(loc='upper left')
    ax.grid()
    plt.tight_layout()

    savestr = '{}_{}h_sz{}_lead{}-{}h.reliabilityCDF.{}.png'.format(
        _site, _interval, _esize, _short, _long, _plot_type.lower())
    print(savestr)

    os.makedirs(_figdir + 'reliabilityCDF/', exist_ok=True)
    plt.savefig(_figdir + 'reliabilityCDF/' + savestr, dpi=150)

    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()
        
def rank_over_leadtime(_data, _interval, _short, _long, _esize, show=False):

    select = _data[((_data['Interval'] == _interval)
                & ((_data['LeadTime'] >= _short) 
                & (_data['LeadTime'] <= _long)))]

    select = select[select['EventSize'] == _esize] if _esize != 'All' else select

    t0, t1 = [threshold_sets[ei][0], threshold_sets[ei][1]] if _esize != 'All' else [0, round(np.nanmax(select['verif_ob']), 2)]

    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), facecolor='w')
    ax3, ax1, ax2 = axs.flatten()
    ms, lw = 50, 1

    for i, lead_time in enumerate(np.unique(select['LeadTime'])):

        lead_data = select[select['LeadTime'] == lead_time]
        lead_data = lead_data.dropna()

        # Lead time vs Mean Rank Verification
        label = 'Mean Verification Rank' if i == 0 else None
        ax1.scatter(lead_time, np.nanmean(lead_data['verif_rank']), c='g', marker='_', s=ms*5, linewidth=lw*2, label=label)

        label = 'Median Verification Rank' if i == 0 else None
        ax1.scatter(lead_time, np.nanpercentile(lead_data['verif_rank'], 50), c='r',marker='_', s=ms*5, linewidth=lw*2, label=label)

        # Lead time vs Mean Deterministic Comparason
        label = 'Mean Deterministic Rank' if i == 0 else None
        ax2.scatter(lead_time, np.nanmean(lead_data['det_rank']), c='g', marker='_', s=ms*5, linewidth=lw*2, label=label)

        label = 'Median Deterministic Rank' if i == 0 else None
        ax2.scatter(lead_time, np.nanpercentile(lead_data['det_rank'], 50), c='r', marker='_', s=ms*5, linewidth=lw*2, label=label)

        ax3.scatter(lead_time, np.nanmean(lead_data['det_fx'] - lead_data['verif_ob']), c='k', marker='_', s=ms*5, linewidth=lw*2)
        # ax3.scatter(lead_time, np.nanmean(lead_data['verif_rank_val'] - lead_data['verif_ob']), c='C0', marker='_', s=ms*5, linewidth=lw*2)
        ax3.scatter(lead_time, np.nanmean(lead_data['mean_fx'] - lead_data['verif_ob']), c='g', marker='_', s=ms*5, linewidth=lw*2)
        ax3.scatter(lead_time, np.nanmean(lead_data['med_fx'] - lead_data['verif_ob']), c='r', marker='_', s=ms*5, linewidth=lw*2)

    ax3.scatter(-10, 0, c='k', marker='_', s=ms*5, linewidth=lw*2, label='Deterministic')
    # ax3.scatter(-10, 0, c='C0', marker='_', s=ms*5, linewidth=lw*2, label='Rank-Matched')
    ax3.scatter(-10, 0, c='g', marker='_', s=ms*5, linewidth=lw*2, label='Mean')
    ax3.scatter(-10, 0, c='r', marker='_', s=ms*5, linewidth=lw*2, label='Median')

    for ax in axs:
        ax.set_xticks(np.arange(_short, _long+1, 12))
        ax.set_xlim([_short-6, _long+2.5])
        ax.set_xticklabels(np.arange(_short, _long+1, 12), rotation=45)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.legend(loc='lower right', fontsize='small')
        ax.grid()
        ax.set_xlim([_short, _long+2.5])

    ax3.set_ylabel('Mean Error (in)\n(Forecast - Observation)\n')

    for ax in [ax3]:
        ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
        ax.set_yticks(np.arange(-.5, .251, .05))
        ax.set_ylim([-.5, .25])
        ax.legend(loc='lower left', fontsize='small')
        ax.set_xlabel('Lead Time')
        
    n_precip_periods = np.unique(select['ValidTime'][~np.isnan(select['verif_ob'])]).shape[0]
    plt.suptitle(('{} Percentile-Matched Verification\nNBM v3.2 {} – {}\n\nEvent Size: {} ({:.2f} – {:.2f} in)\n' + 
                  'Interval: {} h | Lead Time: {} – {} h\nn={}, np={}\n').format(
                _site, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d'),
                _esize, _thresholds[_interval][_thresh_id[_esize][0]], 
                _thresholds[_interval][_thresh_id[_esize][1]],
                _interval, _short, _long, len(select), n_precip_periods))

    plt.tight_layout(rect=[0, 0.03, 1, 0.79])

    savestr = '{}_{}h_sz{}_lead{}-{}h.rank_err_overtime.png'.format(_site, _interval, _esize, _short, _long)
    print(savestr)

    os.makedirs(_figdir + 'rank_err_overtime/', exist_ok=True)
    plt.savefig(_figdir + 'rank_err_overtime/' + savestr, dpi=150)

    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()
        
def get_nbm_1d_mp(_stid, verbose=False):
    
    nbmfile = _datadir + '%s_nbm_%s_%s.pd'%(_stid, _date0.strftime('%Y%m%d'), _date1.strftime('%Y%m%d'))
    
    if os.path.isfile(nbmfile):
        # Load file
        #_nbm = pd.read_pickle(nbmfile)
        if verbose:
            # print('Loaded NBM from file %s'%nbmfile)
            print('NBM file exists%s\n'%nbmfile)
        else:
            pass

    else:
        url_list = []
        for date in _dates:
            for init_hour in _init_hours:
                # For now pull from the csv generator
                # Best to get API access or store locally later
                base = 'https://hwp-viz.gsd.esrl.noaa.gov/wave1d/data/archive/'
                datestr = '{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day)
                sitestr = '/NBM/{:02d}/{:s}.csv'.format(init_hour, _stid)
                url_list.append([date, init_hour, base + datestr + sitestr])
        
        # Try multiprocessing this for speed?
        _nbm = np.array([get_1d_csv(url, this=i+1, total=len(url_list),
                                    verbose=False) for i, url in enumerate(url_list)])
        _nbm = np.array([line for line in _nbm if line is not None])

        try:
            header = _nbm[0, 0]
            
        except:
            if verbose:
                print('No NBM 1D Flat File for %s\n'%_stid)
            _nbm = (_stid, None)
            nbmfile = None
            
        else:
            if verbose:
                print('Producing NBM data from 1D Flat File for %s\n'%_stid)
            # This drops days with incomplete collections. There may be some use
            # to keeping this data, can fix in the future if need be
            # May also want to make the 100 value flexible!
            _nbm = np.array([np.array(line[1]) for line in _nbm if len(line[1]) == 100])

            _nbm = _nbm.reshape(-1, _nbm.shape[-1])
            _nbm[np.where(_nbm == '')] = np.nan

            # Aggregate to a clean dataframe
            _nbm = pd.DataFrame(_nbm, columns=header).set_index(
                ['InitTime', 'ValidTime']).sort_index()

            # Drop last column (misc metadata?)
            _nbm = _nbm.iloc[:, :-2].astype(float)
            header = _nbm.columns

            # variables = np.unique([k.split('_')[0] for k in header])
            # levels = np.unique([k.split('_')[1] for k in header])

            init =  _nbm.index.get_level_values(0)
            valid = _nbm.index.get_level_values(1)

#             # Note the 1h 'fudge factor' in the lead time here
#             lead = pd.DataFrame(
#                 np.transpose([init, valid, ((valid - init).values/3600/1e9).astype(int)+1]), 
#                 columns=['InitTime', 'ValidTime', 'LeadTime']).set_index(['InitTime', 'ValidTime'])

#             _nbm.insert(0, 'LeadTime', lead)

            klist = np.array([k for k in np.unique([k for k in list(_nbm.keys())]) if ('APCP' in k)&('1hr' not in k)])
            klist = klist[np.argsort(klist)]
#             klist = np.append('LeadTime', klist)
            _nbm = _nbm.loc[:, klist]

#             # Nix values where lead time shorter than acc interval
#             for k in _nbm.keys():
#                 if 'APCP24hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 24] = np.nan
#                 elif 'APCP12hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 12] = np.nan
#                 elif 'APCP6hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 6] = np.nan
#                 else:
#                     pass
                
            _nbm.to_pickle(nbmfile)
        
            if verbose:
                print('\nSaved NBM to file %s\n'%nbmfile)

    return nbmfile

def get_precip_obs_mp(_stid, verbose=False):
        
    obfile = _datadir + '%s_obs_%s_%s.pd'%(_stid, _date0.strftime('%Y%m%d'), _date1.strftime('%Y%m%d'))

    if os.path.isfile(obfile):
        if verbose:
            # print('Loaded obs from file %s'%obfile)
            print('Obs File Exists %s'%obfile)
        else:
            pass

    else:
        # Get and save file
        iobs = get_precip_obs(_stid, _date0, _date2, verbose=False)
        iobs = iobs[0].merge(iobs[1], how='inner', on='ValidTime').merge(iobs[2], how='inner', on='ValidTime')
        iobs = iobs[[k for k in iobs.keys() if 'precip' in k]].sort_index()

        iobs.to_pickle(obfile)
        if verbose:
            print('Saved obs to file %s\n'%obfile)
        del iobs
    
    return obfile

def reliability_verif_cdf_multistation(_data, _interval, _short, _long, _plot_type, _plot_var, _esize, show=False):
    import statsmodels.stats.api as sms

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
    
    hist = ax.hist(select[_plot_var], bins=pbins, density=True, cumulative=True,
            color='w', alpha=0, linewidth=3.5, zorder=10)
    histy, histx = hist[0]*100, hist[1][1:]-(pbinsize/2)
    ax.plot(histx, histy, marker='x', linestyle='--', markersize=10, color='k', linewidth=2)
    
    # Plot the spread (compiled max/min from each site - 
    # can use a different metric like 1SD, 90/10, etc if desired)
    hist_spread = []
    select = select.set_index('Site')
    for site in np.unique(select.index):
        hist = ax.hist(select.loc[site][_plot_var], bins=pbins, density=True, cumulative=True,
                color='w', alpha=0, linewidth=3.5, zorder=10)
        histy, histx = hist[0]*100, hist[1][1:]-(pbinsize/2)
        hist_spread.append(histy)
    hist_spread = np.array(hist_spread)
    
    shade_lo = hist_spread.min(axis=0)
    shade_hi = hist_spread.max(axis=0)
    plt.fill_between(histx, shade_lo, shade_hi, color='gray', alpha=0.35, label='Data Range')
    
    conf_alpha = 0.05
    conf = np.array([sms.DescrStatsW(x).tconfint_mean(alpha=conf_alpha) for x in hist_spread.T])
#     ax.fill_between(histx, conf[:, 0], conf[:, 1], color='gray', alpha=0.35,
#                     label='%d/%d CI'%(conf_alpha*100, (1-conf_alpha)*100))
    ax.plot(histx, conf[:, 0], 'r--', alpha=0.85, linewidth=0.8)
    ax.plot(histx, conf[:, 1], 'r--', alpha=0.85, linewidth=0.8, label='5/95 CI')
    
    ax.plot(np.arange(0, 101, 1), np.arange(0, 101, 1), '--k', linewidth=1, zorder=20)

    ax.set_xticks(np.arange(0, 101, 5))
    ax.set_xlim([0, 101])

    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_yticklabels(np.arange(0, 101, 5), rotation=30)
    ax.set_ylim([0, 101])

    ax.set_yticklabels(np.arange(100, -5, -5), rotation=30)

    ax.set_ylabel('\nObserved in % of Forecasts')
    ax.set_xlabel('\nForecast Verifies At/Above Percentile\n')

    n_precip_periods = np.unique(select['ValidTime'][~np.isnan(select['verif_ob'])]).shape[0]
    ax.set_title(('{} Percentile-Matched {}\nNBM v3.2 {} – {}\n\nEvent Size: {} ({:.2f} – {:.2f} in)\n' + 
                  'Interval: {} h | Lead Time: {} – {} h\nn={}, np={}\n').format(
                _site, _plot_type, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d'),
                _esize, _thresholds[_interval][_thresh_id[_esize][0]], 
                _thresholds[_interval][_thresh_id[_esize][1]],
                _interval, _short, _long, len(select), n_precip_periods), size=font_size)

    ax.text(5, 92, 'Wet Bias')
    ax.text(85, 6, 'Dry Bias')
    ax.text(35, 38, 'Unbiased Distribution', rotation=40)

    ax.legend(loc='center left')
    ax.grid()
    plt.tight_layout()

    savestr = '{}_{}h_sz{}_lead{}-{}h.reliabilityCDF.{}.png'.format(
        _site, _interval, _esize, _short, _long, _plot_type.lower())
    print(savestr)

    os.makedirs(_figdir + 'reliabilityCDF/', exist_ok=True)
    plt.savefig(_figdir + 'reliabilityCDF/' + savestr, dpi=150)

    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()