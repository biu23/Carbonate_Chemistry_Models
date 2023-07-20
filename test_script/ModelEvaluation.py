import numpy as np
from cmocean import cm
import cartopy as cp
import cartopy.crs as ccrs
import netCDF4 as nc
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE')
#list of models
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import cartopy.feature as cfeature
from importlib import reload
import matplotlib.path as mpath
import glob
import pickle
import pandas as pd
import seawater
import time
plt.rcParams.update({'font.size': 12})
font = {'family' : 'monospace',
'weight' : 'normal',
'size'   : 12}
plt.rc('font', **font)
import matplotlib
import os
import csv
import argparse

pwd = "/gpfs/home/gdg21vsa/ModelEvaluation/"

# mask and region
regs = ['ARCTIC', 'P1', 'P2', 'P3', 'P4', 'P5', 'A1', 'A2', 'A3', 'A4', 'A5', 'I3', 'I4', 'I5']

regdict = {'ARCTIC' : {'number' : 0.5},
        'P1' : {'number': 1.0},
        'P2' : {'number': 1.2},
        'P3' : {'number': 1.4},
        'P4' : {'number': 1.6},
        'P5' : {'number': 1.8},
            'A1' : {'number': 2.4},
        'A2' : {'number': 2.6},
        'A3' : {'number': 2.8},
        'A4' : {'number': 3},
        'A5' : {'number': 3.2},
        'I3' : {'number': 3.6},
        'I4' : {'number': 3.8},
        'I5' : {'number': 4},
        
        }
    
tics = []
tcm = 'Spectral'
tmask = nc.Dataset('/gpfs/data/greenocean/software/resources/breakdown/clq_basin_masks_ORCA.nc')

maskno = np.zeros([149,182])
for i in range(0, len(regs)):
    maskno[tmask[regs[i]][:] == 1] = regdict[regs[i]]['number']
    tics.append(regdict[regs[i]]['number'])
maskno[maskno == 0] = np.nan


w = plt.pcolor(maskno, cmap = tcm, vmin = 0.5, vmax = 4)
cbar = plt.colorbar(w, ticks=tics)
t = cbar.ax.set_yticklabels(['ARCTIC', 'P1', 'P2', 'P3', 'P4', 'P5', 'A1', 'A2', 'A3', 'A4', 'A5', 'I3', 'I4', 'I5'], fontsize = 9)
plt.suptitle('ocean regions, subdivided')

fact = 0.2
fig = plt.figure(figsize=(30*fact,15*fact))

cmap = matplotlib.cm.get_cmap('Spectral')
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=4)

for i in range(0,len(regs)):
    rgba = cmap(norm(regdict[regs[i]]['number']))
    plt.plot(regdict[regs[i]]['number'], 1, marker = 'o', color = rgba, label = regs[i])
    regdict[regs[i]]['colour'] = rgba
plt.legend(ncol = 5, fontsize = 10)
plt.suptitle('colours assigned')

mean_masks = np.zeros([len(regs), 149, 182])
for i, reg in enumerate(regs):
    mean_masks[i][maskno == regdict[reg]['number']] = 1



def ModelEvaluation (modlist, yrst=1990, yrend=2020):
    
    directory_path = os.path.join(pwd, modlist)

    # create directory, save csv function
    def create_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def save_data_as_csv(data, csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    create_directory(directory_path)

    # loading data
    # GLODAP
    df = pd.read_csv('/gpfs/home/mep22dku/scratch/SOZONE/evalOutput/datasets/GLODAPv2.2022_GLOBAL_valid_DICTA_umolL_STITCHED.csv')
    df = df[(df.YR >= yrst) & (df.YR <= yrend)]
    df = df[df.PRES <= 10]  #surface

    tDIC = np.array(df['DIC'][:])
    tALK = np.array(df['ALK'][:])
    tYEAR = np.array(df['YR'])
    tMONTH = np.array(df['MONTH'])
    tY = np.array(df['Y'])
    tX = np.array(df['X'])

    tREG_new = np.array(tX)

    for j in range(len(tY)):
        y = tY[j].astype(int)
        x = tX[j].astype(int)
        tREG_new[j] = maskno[y, x]

    # model match with obs
    t0 = np.zeros_like(tDIC)
    td = {}
    td = {'YEAR': tYEAR,'MONTH': tMONTH, 'y': tY, 'x': tX ,'obs_DIC':  tDIC, 'obs_TA':tALK, 'REG': tREG_new, 
        'REG_Group': tREG_new}
    tdat_OBS = pd.DataFrame(data=td)
    tdat_OBS = tdat_OBS[(tdat_OBS.obs_DIC>-999) & (tdat_OBS.obs_TA>-999) 
                        & (tdat_OBS.obs_DIC != 0) & (tdat_OBS.obs_TA != 0)]
    
    # read model data
    def make_yearlist(yrst, yrend, dtype, tr, baseDir):
        yrs = np.arange(yrst,yrend+1,1)
        ylist = []
        for i in range(0,len(yrs)):
            ty = f'{baseDir}/{tr}/ORCA2_1m_{yrs[i]}*{dtype}*.nc'
            t2 = glob.glob(ty)
            #print(t2)
            ylist.append(t2[0])
        return ylist

    baseDir = '/gpfs/data/greenocean/software/runs/'
    depth = 0
    dtype = 'ptrc'

    modnam = modlist
    tylist = make_yearlist(yrst, yrend, dtype, modnam, baseDir)
    tdat_MOD = xr.open_mfdataset(tylist)
    
    tdat_MOD_0 = tdat_MOD.isel(deptht= depth)

    # save DIC as Dataframe
    tdat_MOD_DIC = tdat_MOD_0['DIC'].to_dataframe(dim_order=None)
    tdat_MOD_DIC = tdat_MOD_DIC.loc[:, ['DIC']]     #only keep DIC and multiple index
    tdat_MOD_DIC = tdat_MOD_DIC[(tdat_MOD_DIC[['DIC']] != 0).all(axis=1)]
    tdat_MOD_DIC = tdat_MOD_DIC.dropna()

    tdat_MOD_DIC = tdat_MOD_DIC.reset_index()

    tdat_MOD_DIC['YEAR'] = tdat_MOD_DIC['time_counter'].apply(lambda x: x.year)
    tdat_MOD_DIC['MONTH'] = tdat_MOD_DIC['time_counter'].apply(lambda x: x.month)

    # save TA as Dataframe
    tdat_MOD_TA = tdat_MOD_0['Alkalini'].to_dataframe(dim_order=None)
    tdat_MOD_TA = tdat_MOD_TA.loc[:, ['Alkalini']]     #only keep TA and multiple index
    tdat_MOD_TA = tdat_MOD_TA[(tdat_MOD_TA[['Alkalini']] != 0).all(axis=1)]
    tdat_MOD_TA = tdat_MOD_TA.dropna()

    tdat_MOD_TA = tdat_MOD_TA.reset_index()

    tdat_MOD_TA['YEAR'] = tdat_MOD_TA['time_counter'].apply(lambda x: x.year)
    tdat_MOD_TA['MONTH'] = tdat_MOD_TA['time_counter'].apply(lambda x: x.month)

    # merge, match model data and obs data
    tdat_merge_modDIC = pd.merge(tdat_MOD_DIC, tdat_OBS, on=['YEAR', 'MONTH', 'y', 'x'])
    tdat_merge_modDIC = tdat_merge_modDIC.loc[:, ['DIC','obs_DIC', 'REG','REG_Group']]

    tdat_merge_modTA = pd.merge(tdat_MOD_TA, tdat_OBS, on=['YEAR', 'MONTH', 'y', 'x'])
    tdat_merge_modTA = tdat_merge_modTA.loc[:, ['Alkalini','obs_TA','REG','REG_Group']]

    # calculate region mean
    tdat_merge_modDIC_mean = tdat_merge_modDIC.groupby(by=["REG_Group"]).mean()
    
    tdat_merge_modTA_mean = tdat_merge_modTA.groupby(by=["REG_Group"]).mean()

    # plot
    color = ['r','m','c','y','g']
    s = 70 

    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)

    #GLODAP --> after match
    plt.scatter(tdat_merge_modDIC_mean['obs_DIC'],tdat_merge_modDIC_mean['REG'] ,
                marker = 'o', s=s,label = 'GLODAP',color='k', zorder=4)

    # match model data
    plt.scatter(tdat_merge_modDIC_mean['DIC']* 1e6,tdat_merge_modDIC_mean['REG'],
                marker = '+', s=s,label = f'{modlist} match DIC',color='r', zorder=5)

    #background region line
    xmin = 1700
    xmax = 2400
    y = (0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.4, 2.6, 2.8, 3, 3.2, 3.6, 3.8, 4)
    labels = ['ARCTIC', 'P1', 'P2', 'P3', 'P4', 'P5', 'A1', 'A2', 'A3', 'A4', 'A5', 'I3', 'I4', 'I5']

    for i in range(0,len(regs)):
        rgba = cmap(norm(regdict[regs[i]]['number']))
        plt.hlines(y=regdict[regs[i]]['number'], 
                xmin=xmin, 
                xmax=xmax, 
                    colors = rgba, linestyles='--', lw=1, zorder=-4)

        regdict[regs[i]]['colour'] = rgba

    # y label
    plt.yticks(y, labels) #rotation='vertical')

    # legend position
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3) # Put a legend below current axis


    plt.xlabel('DIC, umol/L')
    plt.ylabel('Region')
    plt.title(f'{yrst}-{yrend} Surface DIC')

    image_file_path = os.path.join(directory_path, f'{yrst}-{yrend}Surface_DIC.png')
    plt.savefig(image_file_path)

    # TA plot
    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)

    #GLODAP --> match
    plt.scatter(tdat_merge_modTA_mean['obs_TA'],tdat_merge_modTA_mean['REG'] ,
                marker = 'o', s=s,label = 'GLODAP',color='k', zorder=4)

    # match
    plt.scatter(tdat_merge_modTA_mean['Alkalini']* 1e6,tdat_merge_modTA_mean['REG'] ,
                marker = '+', s=s,label = f'{modlist} match TA',color='r', zorder=5)

    #background region line
    xmin = 1800
    xmax = 2600
    for i in range(0,len(regs)):
        rgba = cmap(norm(regdict[regs[i]]['number']))
        plt.hlines(y=regdict[regs[i]]['number'], 
                xmin=xmin, 
                xmax=xmax, 
                    colors = rgba, linestyles='--', lw=1, zorder=-5)
        
        regdict[regs[i]]['colour'] = rgba

    # y label
    plt.yticks(y, labels) #rotation='vertical')

    # legend position
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3) # Put a legend below current axis

    plt.xlabel('TA, umol/L')
    plt.ylabel('Region')
    plt.title(f'{yrst}-{yrend} Surface TA')

    image_file_path = os.path.join(directory_path, f'{yrst}-{yrend}Surface_TA.png')
    plt.savefig(image_file_path)

    # TA-DIC
    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)

    # GLODAP --> match
    plt.scatter(tdat_merge_modTA_mean['obs_TA']-tdat_merge_modDIC_mean['obs_DIC'],
                tdat_merge_modTA_mean['REG'] ,marker = 'o',s=s, label = 'GLODAP',color='k', zorder=4)


    # match
    plt.scatter((tdat_merge_modTA_mean['Alkalini']-tdat_merge_modDIC_mean['DIC'])* 1e6,
                tdat_merge_modTA_mean['REG'] ,marker = '+', s=s,label = f'{modlist} match TA-DIC',color='r', zorder=5)

    #background region line
    xmin = 0
    xmax = 450
    for i in range(0,len(regs)):
        rgba = cmap(norm(regdict[regs[i]]['number']))
        plt.hlines(y=regdict[regs[i]]['number'], 
                xmin=xmin, 
                xmax=xmax, 
                    colors = rgba, linestyles='--', lw=1, zorder=-5)
        # plt.plot(xmin,regdict[regs[i]]['number'], 
        #          color = rgba, label = regs[i],zorder=-4)        # for legend
        regdict[regs[i]]['colour'] = rgba

    # y label
    plt.yticks(y, labels) #rotation='vertical')

    # legend position
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=3) # Put a legend below current axis


    plt.xlabel('TA-DIC, umol/L')
    plt.ylabel('Region')
    plt.title(f'{yrst}-{yrend} Surface TA-DIC')

    image_file_path = os.path.join(directory_path, f'{yrst}-{yrend}Surface_TA-DIC.png')
    plt.savefig(image_file_path)

    # (2) DATA: selected model evaluation metrics
    def bias_box(obs,mod):
        bias = (-np.mean(obs) + np.mean(mod))
        return bias

    def RMSE_box(obs,mod):
        RMSE = (np.sqrt(np.sum((mod - obs)**2) /len(obs)))
        return RMSE

    def WSS_box(obs,mod):
        xbar = np.mean(obs)
        WSS = (1-(np.sum((mod - obs)**2)  / np.sum((np.abs(mod - xbar) + np.abs(obs - xbar))**2)))
        return WSS
    
    # merge data
    m_obs_DIC = np.array(tdat_merge_modDIC['obs_DIC'][:])
    m_mod_DIC = np.array(tdat_merge_modDIC['DIC'][:])

    m_obs_TA = np.array(tdat_merge_modTA['obs_TA'][:])
    m_mod_TA = np.array(tdat_merge_modTA['Alkalini'][:])


    mask = ~np.isnan(m_mod_DIC) & ~np.isnan(m_mod_TA)
    bias_DIC = bias_box(m_obs_DIC[mask], m_mod_DIC[mask]*1e6)
    RMSE_DIC = RMSE_box(m_obs_DIC[mask], m_mod_DIC[mask]*1e6)
    WSS_DIC = WSS_box(m_obs_DIC[mask], m_mod_DIC[mask]*1e6)

    bias_TA = bias_box(m_obs_TA[mask],m_mod_TA[mask]*1e6)
    RMSE_TA = RMSE_box(m_obs_TA[mask],m_mod_TA[mask]*1e6)
    WSS_TA = WSS_box(m_obs_TA[mask],m_mod_TA[mask]*1e6)

    bias_TADIC = bias_box(m_obs_TA[mask]- m_obs_DIC[mask], (m_mod_TA[mask]-m_mod_DIC[mask])*1e6)
    RMSE_TADIC = RMSE_box(m_obs_TA[mask]- m_obs_DIC[mask], (m_mod_TA[mask]-m_mod_DIC[mask])*1e6)
    WSS_TADIC = WSS_box(m_obs_TA[mask]- m_obs_DIC[mask], (m_mod_TA[mask]-m_mod_DIC[mask])*1e6)

    data_evaluation = [
    {'evaluation_metrics':'bias' ,'DIC': bias_DIC, 'TA': bias_TA, 'TA-DIC': bias_TADIC}, 
    {'evaluation_metrics':'RMSE_DIC' ,'DIC': RMSE_DIC, 'TA': RMSE_TA, 'TA-DIC': RMSE_TADIC},
    {'evaluation_metrics':'WSS' ,'DIC': WSS_DIC,  'TA': WSS_TA,  'TA-DIC': WSS_TADIC}]

    data_evaluation = pd.DataFrame(data_evaluation)

    csv_file_path = os.path.join(directory_path, f'{yrst}-{yrend}_{modlist}_data_evaluation.csv')
    data_evaluation.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Calculating and plotting data using command line arguments')
    parser.add_argument('modlist', type=str, help='one model name')
    parser.add_argument('--yrst',  type=int, default=1990, help='Model data start year, default 1990')
    parser.add_argument('--yrend', type=int, default=2020, help='Model data end year, default 2020')
    
    args = parser.parse_args()
    
    ModelEvaluation(args.modlist, yrst=args.yrst, yrend=args.yrend)










