import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import seaborn as sns
import glob

EGFP_BLEACHING_PARAMS = 9.95346095e-01, 1.26448073e-04
ALEXA647_BLEACHING_PARAMS = 1, 6.45441848e-05


def find_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def transpose_data(df, condition, column):
    rois = np.unique(df['roi'])
    output_df = pd.DataFrame()
    for roi in rois:
        tmpdf=df[df['roi']==roi]
        output_df['roi%d'%roi]=tmpdf.loc[condition, column].values
    return output_df


def plot_egfp_alexa(time, eGFP_df=pd.DataFrame(), alexa_df=pd.DataFrame(), path='', figsize=(7,8)):
    for column in eGFP_df.columns if not eGFP_df.empty else alexa_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        if not eGFP_df.empty:
            ax.plot(time, eGFP_df[column], color='green', label='eGFP')
        if not alexa_df.empty:
            ax.plot(time, alexa_df[column], color='red', label='Alexa 647')
        plt.legend()
        ax.set_title(column)
        fig.savefig(path+column+'.png')
        plt.close()


def calulate_sm(path, time, bcg_fn='bcg1.csv', sample_fn='sample1.csv', Alexa_647=0.325711*0.063851*1.039*1e4,
                eGFP_ch=1, alexa_ch=2, eGFP_sm=828*0.325711, max_limit=15000):
    try:
        os.mkdir('plots')
        os.mkdir('plots/number_of_molecules')
        os.mkdir('plots/alexa_fluo')
        os.mkdir('plots/eGFP_fluo')
        os.mkdir('plots/norm')
    except:
        pass

    os.chdir(path)
    bcg = pd.read_csv(bcg_fn)
    sample = pd.read_csv(sample_fn)
    roinum=len(sample)/(sample['Frame'].max()*sample['Ch'].max())
    bcg['roi']=[int(i%roinum) for i in range(0,len(bcg))]
    sample['roi']=[int(i%roinum) for i in range(0,len(bcg))]
    sample['SubtrInt']=sample['RawIntDen'].values-bcg['RawIntDen'].values
    print("total number of particles selected %d"%len(np.unique(sample['roi'])))
    to_remove=np.unique(sample[sample['Max']>max_limit]['roi'])
    sample=sample[~sample['roi'].isin(to_remove)]
    print("Total number of saturated particles: %d"%len(to_remove))
    print("Removed rois:\n")
    print(to_remove)
    print("\n")
    sample['mol_num']=None
    sample.loc[sample['Ch']==eGFP_ch, 'mol_num']=sample.loc[sample['Ch']==eGFP_ch, 'SubtrInt'].values/eGFP_sm
    sample.loc[sample['Ch']==alexa_ch, 'mol_num']=sample.loc[sample['Ch']==alexa_ch, 'SubtrInt'].values/Alexa_647
    egfp_num=transpose_data(sample, sample['Ch']==eGFP_ch, 'mol_num')
    egfp_num.to_excel('number_of_eGFP_mol.xlsx')
    egfp_fluo=transpose_data(sample,sample['Ch']==eGFP_ch,'SubtrInt')
    egfp_fluo.to_excel('eGFP_fluorescence.xlsx')
    alexa_numm=transpose_data(sample, sample['Ch']==alexa_ch, 'mol_num')
    alexa_numm.to_excel('number_of_Alexa_mol.xlsx')
    alexa_fluo=transpose_data(sample,sample['Ch']==alexa_ch, 'SubtrInt')
    alexa_fluo.to_excel('alexa647_fluorescence.xlsx')

    sample.to_excel('combined_data.xlsx')
    plot_egfp_alexa(time,egfp_num,alexa_numm,'plots/number_of_molecules/')
    plot_egfp_alexa(time, eGFP_df=pd.DataFrame(), alexa_df=alexa_fluo, path='plots/alexa_fluo/')
    plot_egfp_alexa(time, egfp_fluo, path='plots/eGFP_fluo/')

    return sample, egfp_num, egfp_fluo, alexa_numm, alexa_fluo


def load_single_channel_data(path, bcg_fn='bcg1.csv', sample_fn='sample1.csv', max_limit=15000):
    os.chdir(path)
    bcg = pd.read_csv(bcg_fn)
    sample = pd.read_csv(sample_fn)
    roinum=len(sample)/(sample['Slice'].max())
    bcg['roi']=[int(i%roinum) for i in range(0,len(bcg))]
    sample['roi']=[int(i%roinum) for i in range(0,len(bcg))]
    sample['SubtrInt']=sample['RawIntDen'].values-bcg['RawIntDen'].values
    to_remove=np.unique(sample[sample['Max']>max_limit]['roi'])
    sample=sample[~sample['roi'].isin(to_remove)]
    print("Total number of saturated particles: %d"%len(to_remove))
    print("Removed rois:\n")
    print(to_remove)
    print("\n")
    fluo=transpose_data(sample,sample['Max']<max_limit,'SubtrInt')
    return fluo

def del_roi(egfp_num, egfp_fluo, alexa_numm, alexa_fluo, roi_to_del=[]):
    def delete_rois(df, roilist):
        for roi in roilist:
            del df['roi%s'%roi]
        return df

    if roi_to_del==[]:
        return egfp_num, egfp_fluo, alexa_numm, alexa_fluo

    egfp_num=delete_rois(egfp_num,roi_to_del)
    alexa_numm=delete_rois(alexa_numm,roi_to_del)
    alexa_fluo=delete_rois(alexa_fluo,roi_to_del)
    egfp_fluo=delete_rois(egfp_fluo,roi_to_del)
    return egfp_num, egfp_fluo, alexa_numm, alexa_fluo


def normolize_on_gfp(egfp_num, alexa_numm, time, avg_frames=5):
    initial_number=egfp_num.iloc[:avg_frames].mean()
    egfp_norm=pd.DataFrame()
    alexa_norm=pd.DataFrame()
    for column in alexa_numm.columns:
        alexa_norm[column]=alexa_numm[column].values/initial_number[column]
        egfp_norm[column]=egfp_num[column].values/initial_number[column]
    plot_egfp_alexa(time,egfp_norm,alexa_norm,'plots/norm/')
    egfp_norm.to_excel('egfp_norm.xlsx')
    alexa_norm.to_excel('alexa_norm.xlsx')
    return egfp_norm, alexa_norm


def plot_avg(time, egfp_norm, alexa_norm, ylabel="Number of molecules per CENP-C", green_prot_label='Ndc80 bonsai',
             red_prot_label='Mis12'):
    plt.errorbar(time/60, egfp_norm.mean(axis=1), yerr=egfp_norm.sem(axis=1), color='green', label=green_prot_label)
    plt.errorbar(time/60, alexa_norm.mean(axis=1), yerr=alexa_norm.sem(axis=1), color='red', label=red_prot_label)
    plt.ylabel(ylabel)
    plt.xlabel('time min')
    plt.legend()
    return alexa_norm.mean(axis=1), egfp_norm.mean(axis=1)


def files_to_df(file_list):
    res_df=pd.DataFrame()
    for item in file_list:
        tmpdf = pd.read_csv(item)
        tmpdf['File']=item
        tmpdf['roi']=[i for i in range(0, len(tmpdf))]
        res_df = pd.concat([res_df, tmpdf], ignore_index=True)
    return res_df


def loadData(path, removeBcgSat=True, smVal=828, bcgSaturVal=1400):
    os.chdir(path)
    bcg_files = glob.glob('bcg*.csv')
    sample_files = glob.glob('sample*.csv')
    bcg_df=files_to_df(bcg_files)
    sample_df=files_to_df(sample_files)
    if removeBcgSat == True:
        satDf = bcg_df[bcg_df['Max'] >= bcgSaturVal]
        print("Number of removed saturated background removed rois %d" % len(satDf))
        satDf.to_excel('saturated_bcg_rois.xlsx')
        bcg_df=bcg_df[bcg_df['Max'] < bcgSaturVal]
    sample_df['SubtrIntDen'] = sample_df['RawIntDen'].values-bcg_df['RawIntDen'].median()
    print("median background %f" % bcg_df['RawIntDen'].median())
    sample_df['molNum'] = sample_df['SubtrIntDen'].values/smVal
    return sample_df, bcg_df



def correct_photobleaching(data, bleaching_point, params):
    clmns=list(data.columns)
    clmns.remove('Mean')
    clmns.remove('SEM')
    clmns.remove('time')
    corrected_df = data.copy()
    del corrected_df['SEM']
    del corrected_df['Mean']
    # result_index=data['time'].sub(bleaching_point).abs().idxmin()
    func = lambda t, A, k: A*np.exp(-k*(t-bleaching_point))
    for clmn in corrected_df.columns:
        corrected_df.loc[corrected_df['time']>=bleaching_point, clmn] = \
            corrected_df.loc[corrected_df['time']>=bleaching_point, clmn]/\
            func((data.loc[data['time']>=bleaching_point, 'time']*60.),*params)
    del corrected_df['time']
    mean = corrected_df.mean(axis=1)
    sem = corrected_df.sem(axis=1)
    corrected_df['Mean'] = mean
    corrected_df['SEM'] = sem
    corrected_df['time'] = data['time']

    return corrected_df

def determine_particles_size(path, satur_val=15000, bins_num=100, figsize=(10, 7), density=False):
    data, bcg = loadData(path)
    data.to_excel('not_filterd_data.xlsx')
    bcg.to_excel('bcg_data.xlsx')
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    data.hist(column='RawIntDen', bins=bins_num, ax=axs[0, 0], density=density)
    axs[0, 0].set_xlabel('Integral fluorescence a.a.')
    axs[0, 0].set_ylabel('Counts')
    axs[0, 0].set_title('Not filtered data: fluorescence of spots')
    print('Median value of the spot %f' % data['SubtrIntDen'].median())

    data.hist(column='molNum', bins=bins_num, ax=axs[0, 1], density=density)
    axs[0, 1].set_title("Not filtered data:\n Number of eGFP molecules per particle")
    axs[0, 1].set_xlabel('Number of eGFP molecules')
    axs[0, 1].set_ylabel('Counts')
    print('Median nuber of eGFP per spot %f'%data['molNum'].median())

    bcg.hist(column='RawIntDen', bins=bins_num, ax=axs[1, 0], density=density)
    axs[1, 0].set_xlabel('Integral fluorescence a.a.')
    axs[1, 0].set_ylabel('Counts')
    axs[1, 0].set_title('Not filtered data: fluorescence of background')
    print("After data filtration:")
    filtered_data = data[(data['Max'] >= satur_val) | (data['SubtrIntDen'] < 0)]
    print("number of saturated spots %d" % len(filtered_data))
    filtered_data.to_excel("saturated_rois.xlsx")
    filtered_data=data[~((data['Max'] >= satur_val) | (data['SubtrIntDen'] < 0))]
    filtered_data.hist(column='RawIntDen', bins=bins_num, ax=axs[1, 1], density=density)
    axs[1, 1].set_xlabel('Integral fluorescence a.a.')
    axs[1, 1].set_ylabel('Counts')
    axs[1, 1].set_title(' Filtered data: fluorescence of spots')
    print('Median value of the spot %f' % data['SubtrIntDen'].median())
    axs[1, 1].set_xlim(0, 3e5)

    filtered_data.hist(column='molNum', bins=bins_num, ax=axs[1, 2], density=density)
    axs[1, 2].set_title("Filtered data:\n Number of eGFP molecules per particle")
    axs[1, 2].set_xlabel('Number of eGFP molecules')
    axs[1, 2].set_ylabel('Counts')
    print('Median nuber of eGFP per spot %f' % data['molNum'].median())
    filtered_data.to_excel('filtered_data.xlsx')
    axs[0, 2].set_visible(False)
    axs[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axs[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axs[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axs[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axs[0, 2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    return filtered_data, data, axs



def scatter_plots(exp_name, fit_data, egfp_fluo, alexa_fluo, t_wo_ind, final_ind, eGFP_sm=202.65005660920926,
                  Alexa_647=623.44967509, set_limits=None):
    mx_fluo_range=np.max([egfp_fluo.to_numpy().max()/1e4,alexa_fluo.to_numpy().max()/1e4])
    mx_molnum_range=np.max([egfp_fluo.to_numpy().max()/eGFP_sm,alexa_fluo.to_numpy().max()/Alexa_647])
    fluo_range = np.linspace(0,mx_fluo_range+mx_fluo_range*0.1)
    molnum_range = np.linspace(0,mx_molnum_range+mx_molnum_range*0.1)
    initial_egfp=egfp_fluo[:20].mean(axis=0)
    initial_egfp_err=egfp_fluo[:20].sem(axis=0)
    max_alexa=alexa_fluo[t_wo_ind-10:t_wo_ind].mean(axis=0)
    max_alexa_err=alexa_fluo[t_wo_ind-10:t_wo_ind].sem(axis=0)
    max_eGFP=egfp_fluo[t_wo_ind-10:t_wo_ind].mean(axis=0)
    max_eGFP_err=egfp_fluo[t_wo_ind-10:t_wo_ind].sem(axis=0)
    last_alexa=alexa_fluo[final_ind-10:final_ind].mean(axis=0)
    last_alexa_err=alexa_fluo[final_ind-10:final_ind].sem(axis=0)
    last_eGFP=egfp_fluo[final_ind-10:final_ind].mean(axis=0)
    last_eGFP_err=egfp_fluo[final_ind-10:final_ind].sem(axis=0)

    func = lambda x, k, b: k*x+b

    save_df=pd.DataFrame()
    save_df['initial eGFP fluorescence']=initial_egfp
    save_df['initial eGFP fluorescence err']=initial_egfp_err
    save_df['alexa fluorescence before w/o']=max_alexa
    save_df['alexa fluorescence before w/o err']=max_alexa_err
    save_df['eGFP fluorescence before w/o']=max_eGFP
    save_df['eGFP fluorescence before w/o err']=max_eGFP_err
    save_df['eGFP fluorescence after w/o']=last_eGFP
    save_df['eGFP fluorescence after w/o err']=last_eGFP_err
    save_df['alexa fluorescence after w/o']=last_alexa
    save_df['alexa fluorescence after w/o err']=last_alexa_err
    save_df.to_excel('scatter_data.xlsx')


    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(15,10))
############################## FIT DATA AND PLOT FITTINGS ###########################################

    arg, cov = curve_fit(func, initial_egfp/1e4,max_alexa/1e4)
    corr_coef=pearsonr(max_alexa/1e4, func(initial_egfp/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'initial eGFP fluo vs alexa fluo before wo', arg[0], arg[1], corr_coef]
    axs[0,0].plot(fluo_range, func(fluo_range,*arg), color='c', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/1e4,last_alexa/1e4)
    corr_coef=pearsonr(max_alexa/1e4, func(last_alexa/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'initial eGFP fluo vs alexa fluo after wo', arg[0], arg[1], corr_coef]
    axs[0,0].plot(fluo_range, func(fluo_range,*arg), color='m', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/1e4,max_eGFP/1e4)
    corr_coef=pearsonr(max_eGFP/1e4, func(initial_egfp/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'initial eGFP fluo vs eGFP fluo before wo', arg[0], arg[1], corr_coef]
    axs[0,1].plot(fluo_range, func(fluo_range,*arg), color='c', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/1e4,last_eGFP/1e4)
    corr_coef=pearsonr(last_eGFP/1e4, func(initial_egfp/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'initial eGFP fluo vs eGFP fluo after wo', arg[0], arg[1], corr_coef]
    axs[0,1].plot(fluo_range, func(fluo_range,*arg), color='m', lw=0.5)

    arg, cov = curve_fit(func, max_alexa/1e4,max_eGFP/1e4)
    corr_coef=pearsonr(max_eGFP/1e4, func(max_alexa/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'Alexa fluo vs eGFP fluo before wo', arg[0], arg[1], corr_coef]
    axs[0,2].plot(fluo_range, func(fluo_range,*arg), color='c', lw=0.5)

    arg, cov = curve_fit(func, last_alexa/1e4,last_eGFP/1e4)
    corr_coef=pearsonr(last_eGFP/1e4, func(last_alexa/1e4, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'Alexa fluo vs eGFP fluo after wo', arg[0], arg[1], corr_coef]
    axs[0,2].plot(fluo_range, func(fluo_range,*arg), color='m', lw=0.5)


    arg, cov = curve_fit(func, initial_egfp/eGFP_sm,max_alexa/Alexa_647)
    corr_coef=pearsonr(max_alexa/Alexa_647, func(initial_egfp/eGFP_sm, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'CENP-C vs Mis12 before wo', arg[0], arg[1], corr_coef]
    axs[1,0].plot(molnum_range, func(molnum_range,*arg), color='c', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/eGFP_sm,last_alexa/Alexa_647)
    corr_coef=pearsonr(max_alexa/Alexa_647, func(last_alexa/Alexa_647, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'CENP-C vs Mis12 after wo', arg[0], arg[1], corr_coef]
    axs[1,0].plot(molnum_range, func(molnum_range,*arg), color='m', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/eGFP_sm,max_eGFP/eGFP_sm)
    corr_coef=pearsonr(max_eGFP/eGFP_sm, func(initial_egfp/eGFP_sm, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'CENP-C vs Ndc80 before wo', arg[0], arg[1], corr_coef]
    axs[1,1].plot(molnum_range, func(molnum_range,*arg), color='c', lw=0.5)

    arg, cov = curve_fit(func, initial_egfp/eGFP_sm,last_eGFP/eGFP_sm)
    corr_coef=pearsonr(last_eGFP/eGFP_sm, func(initial_egfp/eGFP_sm, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'CENP-C vs Ndc80 after wo', arg[0], arg[1], corr_coef]
    axs[1,1].plot(molnum_range, func(molnum_range,*arg), color='m', lw=0.5)

    arg, cov = curve_fit(func, max_alexa/Alexa_647,max_eGFP/eGFP_sm)
    corr_coef=pearsonr(max_eGFP/eGFP_sm, func(max_alexa/Alexa_647, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'Mis12 vs Ndc80 before wo', arg[0], arg[1], corr_coef]
    axs[1,2].plot(molnum_range, func(molnum_range,*arg), color='m', lw=0.5)

    arg, cov = curve_fit(func, last_alexa/Alexa_647,last_eGFP/eGFP_sm)
    corr_coef=pearsonr(last_eGFP/eGFP_sm, func(last_alexa/Alexa_647, *arg))
    fit_data.loc[len(fit_data.index)]=[exp_name, 'Mis12 vs Ndc80 after wo', arg[0], arg[1], corr_coef]
    axs[1,2].plot(molnum_range, func(molnum_range,*arg), color='c', lw=0.5)

    ################################ PLOT SCATTER ###########################################

    axs[0, 0].errorbar(initial_egfp/1e4,max_alexa/1e4,xerr=initial_egfp_err/1e4,yerr=max_alexa_err/1e4,ls='',
                       color='c', label='befor washout')
    axs[0, 0].errorbar(initial_egfp/1e4,last_alexa/1e4,xerr=initial_egfp_err/1e4,yerr=last_alexa_err/1e4,ls='',
                       color='m', label='after washout')
    axs[0, 0].set_xlabel('initial eGFP fluorescence $10^4$ a.u.')
    axs[0, 0].set_ylabel('alexa fluorescence $10^4$ a.u. ')


    axs[0, 1].errorbar(initial_egfp/1e4,max_eGFP/1e4,xerr=initial_egfp_err/1e4,yerr=max_eGFP_err/1e4,ls='',
                       label='before washout', color='c')
    axs[0, 1].errorbar(initial_egfp/1e4,last_eGFP/1e4,xerr=initial_egfp_err/1e4,yerr=last_eGFP_err/1e4,ls='',
                       label='after washout', color='m')
    axs[0, 1].set_xlabel('initial eGFP fluorescence $10^4$ a.u.')
    axs[0, 1].set_ylabel('eGFP fluorescence $10^4$ a.u. ')

    axs[0,2].errorbar(max_alexa/1e4,max_eGFP/1e4,xerr=max_alexa_err/1e4,yerr=max_eGFP_err/1e4,ls='',
                      label='before washout', color='c')
    axs[0,2].errorbar(last_alexa/1e4,last_eGFP/1e4,xerr=last_alexa_err/1e4,yerr=last_eGFP_err/1e4,ls='',
                      label='before washout', color='m')
    axs[0,2].set_xlabel('alexa fluorescence $10^4$ a.u.')
    axs[0,2].set_ylabel('eGFP fluorescence $10^4$ a.u.')

    ########### RECALCULATED TO NUMBER OF MOLECULES ########################

    axs[1, 0].errorbar(initial_egfp/eGFP_sm,max_alexa/Alexa_647,xerr=initial_egfp_err/eGFP_sm,
                       yerr=max_alexa_err/Alexa_647,ls='', color='c', label='befor washout')
    axs[1, 0].errorbar(initial_egfp/eGFP_sm,last_alexa/Alexa_647,xerr=initial_egfp_err/eGFP_sm,
                       yerr=last_alexa_err/Alexa_647,ls='', color='m', label='after washout')
    axs[1, 0].set_xlabel('Number of\n CENP-C molecules')
    axs[1, 0].set_ylabel('Number of\n Mis12/KNL1 molecules')

    axs[1, 1].errorbar(initial_egfp/eGFP_sm,max_eGFP/eGFP_sm,xerr=initial_egfp_err/eGFP_sm,
                       yerr=max_eGFP_err/eGFP_sm,ls='', label='before washout', color='c')
    axs[1, 1].errorbar(initial_egfp/eGFP_sm,last_eGFP/eGFP_sm,xerr=initial_egfp_err/eGFP_sm,
                       yerr=last_eGFP_err/eGFP_sm,ls='', label='after washout', color='m')
    axs[1, 1].set_xlabel('Number of\n CENP-C molecules')
    axs[1, 1].set_ylabel('Number of\n Ndc80 molecules')

    axs[1,2].errorbar(max_alexa/Alexa_647,max_eGFP/eGFP_sm,xerr=max_alexa_err/Alexa_647,
                      yerr=max_eGFP_err/eGFP_sm,ls='', label='before washout', color='c')
    axs[1,2].errorbar(last_alexa/Alexa_647,last_eGFP/eGFP_sm,xerr=last_alexa_err/Alexa_647,
                      yerr=last_eGFP_err/eGFP_sm,ls='', label='before washout', color='m')
    axs[1,2].set_xlabel('Number of\n Mis12/KNL1 molecules')
    axs[1,2].set_ylabel('Number of\n Ndc80 molecules')

    if set_limits==None:
        for i in range(0,3):
            axs[0,i].set_xlim(fluo_range.min(), fluo_range.max())
            axs[0,i].set_ylim(fluo_range.min(), fluo_range.max())
            axs[1,i].set_xlim(molnum_range.min(),molnum_range.max())
            axs[1,i].set_ylim(molnum_range.min(),molnum_range.max())
    else:
        set_limits(axs)
    plt.tight_layout()
    return fit_data
