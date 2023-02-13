import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle
def maximum_value_in_column(column):    
    highlight = 'font-weight: bold;'
    default = ''
    maximum_in_column = column.max()
    return [highlight if v == maximum_in_column else default for v in column]

def min_value_in_column(column):    
    highlight = 'font-weight: bold;'
    default = ''
    min_in_column = column.min()
    return [highlight if v == min_in_column else default for v in column]

def parse_synthetic_results(problem_map, save_dict=False):
    exp_dict = {'surrogate': [], 'acquisition': [], 'seed':[], 'problem_name':[], 'problem_idx':[], 'dim':[], 'data':[], 'dist_nearest_train':[], 'inst_regret_test':[], 'inst_regret_pool':[], 'tot_regret_test':[], 'tot_regret_pool':[], 'calibration_mse':[], 'sharpness':[], 'x_opt_dist_test':[], 'x_opt_dist_pool':[], 'path':[]}
    main_directory="./results_synth_data/"
    for foldername in os.listdir(main_directory):
        folder = os.path.join(main_directory, foldername)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.find("parameters") != -1:
                    json_file = open(os.path.join(folder, filename))
                    params = json.load(json_file)
                elif filename.find("metrics") != -1:
                    json_file = open(os.path.join(folder, filename))
                    metrics = json.load(json_file)
            if params['bo'] == True and params['recalibrate']==False:
                exp_dict['surrogate'].append(params['surrogate'])
                exp_dict['acquisition'].append(params['acquisition'])
                exp_dict['data'].append(params['data_name'])
                exp_dict['problem_idx'].append(params['problem_idx'])
                exp_dict['dim'].append(params['d'])
                exp_dict['problem_name'].append(problem_map[str(params['d'])][params['problem_idx']])
                exp_dict['dist_nearest_train'].append(np.array(metrics['next_sample_train_distance']))
                exp_dict['inst_regret_test'].append(np.array(metrics['y_regret_test']))
                exp_dict['inst_regret_pool'].append(np.array(metrics['y_regret_pool']))
                exp_dict['tot_regret_test'].append(np.cumsum(metrics['y_regret_test']))
                exp_dict['tot_regret_pool'].append(np.cumsum(metrics['y_regret_pool']))
                exp_dict['x_opt_dist_pool'].append(np.array(metrics['x_y_opt_dist_pool']))
                exp_dict['x_opt_dist_test'].append(np.array(metrics['x_y_opt_dist_test']))
                exp_dict['sharpness'].append(np.array(metrics['mean_sharpness']))
                exp_dict['calibration_mse'].append(np.array(metrics['y_calibration_mse']))
                exp_dict['seed'].append(params['seed'])
                exp_dict['path'].append(folder)
    if save_dict:
        with open('parsed_results/synth_results.obj', 'wb') as save_file:
            pickle.dump(exp_dict, save_file)
    df = pd.DataFrame.from_dict(exp_dict)
    return df

def aggregate_synthetic_results(df):
    processed_results = {'surrogate': [], 'acquisition': [], 'problem_name':[], 'seed':[], 'problem_idx':[], 'dim':[], 'data':[], 'dist_nearest_train':[], 'inst_regret_test':[], 'inst_regret_pool':[], 'tot_regret_test':[], 'tot_regret_pool':[], 'calibration_mse':[], 'sharpness':[], 'x_opt_dist_test':[], 'x_opt_dist_pool':[]}
    for index, row in df.iterrows():
        processed_results['surrogate'].append(row['surrogate'])
        processed_results['acquisition'].append(row['acquisition'])
        processed_results['data'].append(row['data'])
        processed_results['seed'].append(row['seed'])
        processed_results['problem_idx'].append(row['problem_idx'])
        processed_results['problem_name'].append(row['problem_name'])
        processed_results['dim'].append(row['dim'])
        processed_results['dist_nearest_train'].append(np.array(row['dist_nearest_train']).astype(float).mean()) #Mean across one BO run.
        processed_results['inst_regret_pool'].append(np.array(row['inst_regret_pool'])[-1]) #Last iter.
        processed_results['inst_regret_test'].append(np.array(row['inst_regret_test'])[-1]) #Last iter.
        processed_results['tot_regret_test'].append(np.array(row['tot_regret_test'])[-1]) #Last iter.
        processed_results['tot_regret_pool'].append(np.array(row['tot_regret_pool'])[-1]) #Last iter.
        processed_results['calibration_mse'].append(np.array(row['calibration_mse']).mean(axis=0)) #Mean Calibration MSE over run.
        processed_results['sharpness'].append(np.array(row['sharpness']).mean()) #Mean sharpness over run.
        processed_results['x_opt_dist_test'].append(np.array(row['x_opt_dist_test'])[-1])
        processed_results['x_opt_dist_pool'].append(np.array(row['x_opt_dist_pool'])[-1])
    df = pd.DataFrame.from_dict(processed_results)
    return df

def BO_performance_table1_synth_results(df):
    acqs = list(set(df['acquisition']))
    surrogates = list(set(df['surrogate']))
    seeds = len(list(set(df['seed'])))
    aggregated_processed_results = {'surrogate': [], 'acquisition': [], 'inst_regret_pool':[], 'inst_regret_pool_ste':[], 'tot_regret_pool':[], 'tot_regret_pool_ste':[], 'calibration_mse':[], 'calibration_mse_ste':[], 'sharpness':[], 'sharpness_ste':[]}
    #Aggregate results for pairs of surrogates and acquisitions.
    for acq in acqs:
        for surrogate in surrogates:
            selection = df.loc[((df['acquisition']==acq) & (df['surrogate']==surrogate))]
            aggregated_processed_results['surrogate'].append(surrogate)
            aggregated_processed_results['acquisition'].append(acq)
            aggregated_processed_results['inst_regret_pool'].append(np.mean(selection['inst_regret_pool']))
            aggregated_processed_results['inst_regret_pool_ste'].append(np.std(selection['inst_regret_pool'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['tot_regret_pool'].append(np.mean(selection['tot_regret_pool']))
            aggregated_processed_results['tot_regret_pool_ste'].append(np.std(selection['tot_regret_pool'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['calibration_mse'].append(np.mean(selection['calibration_mse']))
            aggregated_processed_results['calibration_mse_ste'].append(np.std(selection['calibration_mse'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['sharpness'].append(np.mean(selection['sharpness']))
            aggregated_processed_results['sharpness_ste'].append(np.std(selection['sharpness'])/np.sqrt(len(selection['surrogate'])))
    #Aggregate results of surrogates without including random sampling acquisition.
    for surrogate in surrogates:
        selection = df.loc[(df['surrogate']==surrogate) & (df['acquisition'] != "RS")]
        aggregated_processed_results['surrogate'].append(surrogate)
        aggregated_processed_results['acquisition'].append("avg")
        aggregated_processed_results['inst_regret_pool'].append(np.mean(selection['inst_regret_pool']))
        aggregated_processed_results['inst_regret_pool_ste'].append(np.std(selection['inst_regret_pool'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['tot_regret_pool'].append(np.mean(selection['tot_regret_pool']))
        aggregated_processed_results['tot_regret_pool_ste'].append(np.std(selection['tot_regret_pool'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['calibration_mse'].append(np.mean(selection['calibration_mse']))
        aggregated_processed_results['calibration_mse_ste'].append(np.std(selection['calibration_mse'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['sharpness'].append(np.mean(selection['sharpness']))
        aggregated_processed_results['sharpness_ste'].append(np.std(selection['sharpness'])/np.sqrt(len(selection['surrogate'])))
    df = pd.DataFrame.from_dict(aggregated_processed_results)
    df = df.round(4)
    latex_df = pd.DataFrame(columns=['Surrogate', 'Acquisition', 'Inst. Regret', 'Total Regret', 'Calibration Error', 'Sharpness'])
    latex_df['Surrogate'] = df['surrogate']
    latex_df['Acquisition'] = df['acquisition']
    latex_df['Inst. Regret'] = df['inst_regret_pool'].astype(str) + " \pm " + df['inst_regret_pool_ste'].astype(str)
    latex_df['Total Regret'] = df['tot_regret_pool'].astype(str) + " \pm " + df['tot_regret_pool_ste'].astype(str)
    latex_df['Calibration Error'] = df['calibration_mse'].astype(str) + " \pm " + df['calibration_mse_ste'].astype(str)
    latex_df['Sharpness'] = df['sharpness'].astype(str) + " \pm " + df['sharpness_ste'].astype(str)
    table_with_highlight = df.style.apply(min_value_in_column, subset=['inst_regret_pool', "tot_regret_pool", 'calibration_mse'], axis=0)
    return table_with_highlight, latex_df

def parse_real_results(save_dict=False, main_directory=None, recal=False):
    exp_dict = {'surrogate': [], 'acquisition': [], 'seed':[], 'data':[], 'dist_nearest_train':[], 'inst_regret_test':[], 'inst_regret_pool':[], 'tot_regret_test':[], 'tot_regret_pool':[], 'std_change':[], 'calibration_mse':[], 'sharpness':[], 'x_opt_dist_test':[], 'x_opt_dist_pool':[], 'elpd':[]}
    if main_directory is None:
        main_directory="./results_real_data/"
    subdirectories = ['results_FashionMNIST/', "results_FashionMNIST_CNN/", 'results_mnist/', 'results_MNIST_CNN/', 'results_News/', 'results_SVM/']
    for subdirectory in subdirectories:
        full_path = os.path.join(main_directory, subdirectory)
        for foldername in os.listdir(full_path):
            folder = os.path.join(full_path, foldername)
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    if filename.find("parameters") != -1:
                        json_file = open(os.path.join(folder, filename))
                        params = json.load(json_file)
                    elif filename.find("metrics") != -1:
                        json_file = open(os.path.join(folder, filename))
                        metrics = json.load(json_file)
                if params['bo'] == True and params['recalibrate'] == recal:
                    exp_dict['surrogate'].append(params['surrogate'])
                    exp_dict['acquisition'].append(params['acquisition'])
                    exp_dict['data'].append(params['data_name'])
                    exp_dict['std_change'].append(params['std_change'])
                    exp_dict['dist_nearest_train'].append(metrics['next_sample_train_distance'])
                    exp_dict['inst_regret_test'].append(metrics['y_regret_test'])
                    exp_dict['inst_regret_pool'].append(metrics['y_regret_pool'])
                    exp_dict['tot_regret_test'].append(np.cumsum(metrics['y_regret_test']))
                    exp_dict['tot_regret_pool'].append(np.cumsum(metrics['y_regret_pool']))
                    exp_dict['x_opt_dist_pool'].append(metrics['x_y_opt_dist_pool'])
                    exp_dict['x_opt_dist_test'].append(metrics['x_y_opt_dist_test'])
                    exp_dict['sharpness'].append(metrics['mean_sharpness'])
                    exp_dict['calibration_mse'].append(metrics['y_calibration_mse'])
                    exp_dict['seed'].append(params['seed'])
                    exp_dict['elpd'].append(metrics['elpd'])
    df = pd.DataFrame.from_dict(exp_dict)
    if save_dict:
        if recal:
            with open('parsed_results/real_results_recal.obj', 'wb') as save_file:
                pickle.dump(exp_dict, save_file)
        else:
            with open('parsed_results/real_results.obj', 'wb') as save_file:
                pickle.dump(exp_dict, save_file)   
    return df

def aggregate_real_results(df):
    processed_results = {'surrogate': [], 'acquisition': [], 'seed':[], 'data':[], 'dist_nearest_train':[], 'inst_regret_test':[], 'inst_regret_pool':[], 'tot_regret_test':[], 'tot_regret_pool':[], 'std_change':[], 'calibration_mse':[], 'sharpness':[], 'x_opt_dist_test':[], 'x_opt_dist_pool':[], 'elpd':[]}
    for index, row in df.iterrows():
        processed_results['surrogate'].append(row['surrogate'])
        processed_results['acquisition'].append(row['acquisition'])
        processed_results['data'].append(row['data'])
        processed_results['seed'].append(row['seed'])
        processed_results['std_change'].append(row['std_change'])
        processed_results['dist_nearest_train'].append(np.array(row['dist_nearest_train']).astype(float).mean()) #Mean across one BO run.
        processed_results['inst_regret_pool'].append(np.array(row['inst_regret_pool'])[-1]) #Last iter.
        processed_results['inst_regret_test'].append(np.array(row['inst_regret_test'])[-1]) #Last iter.
        processed_results['tot_regret_test'].append(np.array(row['tot_regret_test'])[-1]) #Last iter.
        processed_results['tot_regret_pool'].append(np.array(row['tot_regret_pool'])[-1]) #Last iter.
        processed_results['calibration_mse'].append(np.array(row['calibration_mse']).mean(axis=0)) #Mean Calibration MSE over run.
        processed_results['sharpness'].append(np.array(row['sharpness']).mean()) #Mean sharpness over run.
        processed_results['x_opt_dist_test'].append(np.array(row['x_opt_dist_test'])[-1])
        processed_results['x_opt_dist_pool'].append(np.array(row['x_opt_dist_pool'])[-1])
        processed_results['elpd'].append(row['elpd'][-1]) #Last iteration ELPD.
    df = pd.DataFrame.from_dict(processed_results)
    return df

def BO_performance_table1_real_results(df):
    acqs = list(set(df['acquisition']))
    surrogates = list(set(df['surrogate']))
    aggregated_processed_results = {'surrogate': [], 'acquisition': [], 'inst_regret_pool':[], 'inst_regret_pool_ste':[], 'tot_regret_pool':[], 'tot_regret_pool_ste':[], 'calibration_mse':[], 'calibration_mse_ste':[], 'sharpness':[], 'sharpness_ste':[]}
    #Aggregate results for pairs of surrogates and acquisitions.
    for acq in acqs:
        for surrogate in surrogates:
            selection = df.loc[((df['acquisition']==acq) & (df['surrogate']==surrogate))]
            aggregated_processed_results['surrogate'].append(surrogate)
            aggregated_processed_results['acquisition'].append(acq)
            aggregated_processed_results['inst_regret_pool'].append(np.mean(selection['inst_regret_pool']))
            aggregated_processed_results['inst_regret_pool_ste'].append(np.std(selection['inst_regret_pool'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['tot_regret_pool'].append(np.mean(selection['tot_regret_pool']))
            aggregated_processed_results['tot_regret_pool_ste'].append(np.std(selection['tot_regret_pool'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['calibration_mse'].append(np.mean(selection['calibration_mse']))
            aggregated_processed_results['calibration_mse_ste'].append(np.std(selection['calibration_mse'])/np.sqrt(len(selection['surrogate'])))
            aggregated_processed_results['sharpness'].append(np.mean(selection['sharpness']))
            aggregated_processed_results['sharpness_ste'].append(np.std(selection['sharpness'])/np.sqrt(len(selection['surrogate'])))
    #Aggregate results of surrogates without including random sampling acquisition.
    for surrogate in surrogates:
        selection = df.loc[(df['surrogate']==surrogate) & (df['acquisition'] != "RS")]
        aggregated_processed_results['surrogate'].append(surrogate)
        aggregated_processed_results['acquisition'].append("avg")
        aggregated_processed_results['inst_regret_pool'].append(np.mean(selection['inst_regret_pool']))
        aggregated_processed_results['inst_regret_pool_ste'].append(np.std(selection['inst_regret_pool'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['tot_regret_pool'].append(np.mean(selection['tot_regret_pool']))
        aggregated_processed_results['tot_regret_pool_ste'].append(np.std(selection['tot_regret_pool'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['calibration_mse'].append(np.mean(selection['calibration_mse']))
        aggregated_processed_results['calibration_mse_ste'].append(np.std(selection['calibration_mse'])/np.sqrt(len(selection['surrogate'])))
        aggregated_processed_results['sharpness'].append(np.mean(selection['sharpness']))
        aggregated_processed_results['sharpness_ste'].append(np.std(selection['sharpness'])/np.sqrt(len(selection['surrogate'])))
    df = pd.DataFrame.from_dict(aggregated_processed_results)
    df = df.round(5)
    df = df.sort_values(by=['acquisition', 'surrogate'])
    latex_df = pd.DataFrame(columns=['Surrogate', 'Acquisition', 'Inst. Regret', 'Total Regret', 'Calibration Error', 'Sharpness'])
    latex_df['Surrogate'] = df['surrogate']
    latex_df['Acquisition'] = df['acquisition']
    latex_df['Inst. Regret'] = df['inst_regret_pool'].astype(str) + " \pm " + df['inst_regret_pool_ste'].astype(str)
    latex_df['Total Regret'] = df['tot_regret_pool'].astype(str) + " \pm " + df['tot_regret_pool_ste'].astype(str)
    latex_df['Calibration Error'] = df['calibration_mse'].astype(str) + " \pm " + df['calibration_mse_ste'].astype(str)
    latex_df['Sharpness'] = df['sharpness'].astype(str) + " \pm " + df['sharpness_ste'].astype(str)
    table_with_highlight = df.style.apply(min_value_in_column, subset=['inst_regret_pool', "tot_regret_pool", 'calibration_mse'], axis=0)
    return table_with_highlight, latex_df