import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Union
from torchmetrics import F1Score, Precision, Recall, Accuracy
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import json
import os
import re
import argparse
import json
import logging
import h5py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def read_csv_robust(filepath, **kwargs):
    try:
        return pd.read_csv(filepath, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='latin1', **kwargs)

def extract_csv(original_csv_file: str, target_outcome: list[str], composite_outcome: bool = True, imputing_method: str = "censor_date", demographic_csv_file: Optional[str] = None, demographic_stratification_var: Optional[list[str]] = None):
    """
    Extract the target outcome from the original CSV file.
    
    Parameters:
    - original_csv_file: Path to the original CSV file.
    - target_outcome: List of target outcome columns. By construction, the target outcome must be binary (0/1); no NaN inside the csv for outcomes
    - composite_outcome: Whether to consider the composite outcome or not (at least one of the target outcome must be present to consider the subject as having the outcome)
    - imputing_method: Method to impute the time with censdate (options: "remove_events", "censor_date", "mask_events")
    - demographic_csv_file: Path to the demographic CSV file
    - demographic_stratification_var: List of demographic variables to use for stratification
    
    Returns:
    - A pandas DataFrame containing the subject with target outcome and corresponding time. If demographic_csv_file is provided, also returns the demographic data (age and gender)
    """

    assert imputing_method in ("remove_events", "censor_date", "mask_events")

    df = read_csv_robust(original_csv_file)

    result_df = pd.DataFrame()
    result_df['id'] = df['nsrrid']

    if isinstance(target_outcome, str):
        target_outcome = [target_outcome]

    if target_outcome == ["all"]:
        target_outcome = [col for col in df.columns if "date" not in col and col != "nsrrid"]
    elif "not_" in target_outcome[0]:
        outcome_to_exclude = re.search("not_(.*)", target_outcome[0]).group(1)
        target_outcome = [col for col in df.columns if "date" not in col and col != "nsrrid" and col != target_outcome[0].replace("not_", "")]

    assert all([out in df.columns for out in target_outcome]), f"Target outcome {target_outcome} not found in the CSV file"
    
    # Outcomes > 1 AND Composite False
    if len(target_outcome) > 1 and not composite_outcome:
        mask_keep_global = pd.Series(True, index=df.index)
        for outcome in target_outcome:
            outcome_name = outcome
            time_name = outcome_name + '_date' 

            final_outcome = df[outcome_name].copy()
            final_time = df['censdate'].copy()
            
            # If event=1, time = event_time
            is_event = final_outcome == 1
            event_times = df[time_name]
            
            # Update time where event=1 and time is not NaN
            valid_times = is_event & event_times.notna()
            final_time[valid_times] = event_times[valid_times]

            nan_mask = is_event & event_times.isna()
            if imputing_method == "remove_events":
                mask_keep_global[nan_mask] = False
            elif imputing_method == "censor_date":
                final_outcome[nan_mask] = 1
                final_time[nan_mask] = df.loc[nan_mask, 'censdate'].values
            elif imputing_method == "mask_events":
                final_outcome[nan_mask] = -1
                final_time[nan_mask] = -1
            
            result_df[outcome_name] = final_outcome.astype('Int64')
            result_df[outcome_name + '_time'] = final_time
        
        result_df = result_df[mask_keep_global]
    
    # Calculate Event and Time
    elif len(target_outcome) == 1:
        # Single outcome case
        outcome_name = target_outcome[0]
        time_name = outcome_name + '_date'
        
        final_outcome = df[outcome_name].copy()
        final_time = df['censdate'].copy()
        
        # If event=1, time = event_time
        is_event = final_outcome == 1
        event_times = df[time_name]
        
        # Update time where event=1 and time is not NaN
        valid_times = is_event & event_times.notna()
        final_time[valid_times] = event_times[valid_times]
        
        # NaN Handling
        keep_mask = pd.Series(True, index=df.index)
        nan_mask = is_event & event_times.isna()
        if imputing_method == "remove_events":
            keep_mask[nan_mask] = False
        elif imputing_method == "censor_date":
            final_outcome[nan_mask] = 1
            final_time[nan_mask] = df.loc[nan_mask, 'censdate'].values
        elif imputing_method == "mask_events":
            final_outcome[nan_mask] = -1
            final_time[nan_mask] = -1
        
        result_df['outcome'] = final_outcome.astype('Int64')
        result_df['time'] = final_time
        result_df = result_df[keep_mask]
            
    else:
        # Composite True (and len > 1)
        # 1. Identify events
        events_mask = (df[target_outcome] == 1).any(axis=1)
        
        # 2. Calculate Composite Time
        time_cols_names = [col + '_date' for col in target_outcome]
        times_df = df[time_cols_names].copy()
        relevant_times = times_df.where(df[target_outcome] == 1)

        if imputing_method == "censor_date":
            for col in target_outcome:
                # Check if this specific outcome occurred but has no date
                mask_impute = (df[col] == 1) & (relevant_times[col].isna())
                relevant_times.loc[mask_impute, col] = df.loc[mask_impute, 'censdate']
        min_event_time = relevant_times.min(axis=1)
        
        final_outcome = pd.Series(0, index=df.index)
        final_time = df['censdate'].copy()
        
        final_outcome[events_mask] = 1
        valid_event_times = events_mask & min_event_time.notna()
        final_time[valid_event_times] = min_event_time[valid_event_times]
        
        # 3. Filtering (NaN Handling)
        keep_mask = pd.Series(True, index=df.index)
        
        any_nan_event = pd.Series(False, index=df.index)
        for col in target_outcome:
            t_col = col + '_date'
            # Check if this specific outcome occurred but has no date
            cond = (df[col] == 1) & (df[t_col].isna())
            any_nan_event = any_nan_event | cond
        if imputing_method == "remove_events":
            keep_mask[any_nan_event] = False
        elif imputing_method == "censor_date":
            final_outcome[any_nan_event] = 1
            final_time[any_nan_event] = df.loc[any_nan_event, 'censdate'].values
        elif imputing_method == "mask_events":
            final_outcome[any_nan_event] = -1
            final_time[any_nan_event] = -1

        result_df['outcome'] = final_outcome.astype('Int64')
        result_df['time'] = final_time
        result_df = result_df[keep_mask]
            
    add_medications_col = False

    if demographic_csv_file and demographic_stratification_var:
        if "med" in demographic_stratification_var:
            demographic_stratification_var_filt = demographic_stratification_var.copy()
            demographic_stratification_var_filt.remove("med")
            add_medications_col = True
        else:
            demographic_stratification_var_filt = demographic_stratification_var.copy()
            add_medications_col = False
        demographic_df = read_csv_robust(demographic_csv_file)
        columns_demographic = []
        for keyword in ['nsrrid'] + demographic_stratification_var_filt:
            matches = [col for col in demographic_df.columns if keyword.lower() in col.lower()]
            columns_demographic.extend(matches)
        n_exam = re.search(r'shhs(\d+)', os.path.basename(Path(demographic_csv_file).stem)).group(1)
        columns_demographic = [col for col in columns_demographic if re.search(r'(\d+)', col) is None or re.search(r'(\d+)', col).group() == n_exam]
        demographic_df = demographic_df[columns_demographic]
        demographic_df.rename(columns={'nsrrid': 'id'}, inplace=True)
        demographic_df.rename(columns={col: col.replace(n_exam, '') for col in demographic_df.columns}, inplace=True)
        result_df = pd.merge(result_df, demographic_df, on='id', how='inner')
        for col in demographic_df:
            result_df[col] = result_df[col].fillna(np.median(result_df[col].dropna()))

    if add_medications_col:
        result_df = aggregate_medications_col(result_df, demographic_csv_file)

    return result_df.reset_index(drop=True).convert_dtypes()

def aggregate_medications_col(df, csv_file):
    demographic_df = read_csv_robust(csv_file)
    demographic_df_filtered = demographic_df[demographic_df['nsrrid'].isin(df['id'])]

    # Cardio medications
    medications_list_cardio = ["a2ad", "aced", "alpha", "alphad", "asa", "beta", "betad", "loop", "vasod", "diuret", "diur"]
    regex_pattern = f"^({'|'.join(medications_list_cardio)})(1|2)?$"
    matching_columns = [col for col in demographic_df_filtered.columns if re.match(regex_pattern, col)]
    df_medications_cardio = demographic_df_filtered[['nsrrid'] + matching_columns]
    # keep the max of the medications col list
    df_medications_cardio['med_c'] = df_medications_cardio[matching_columns].max(axis=1).fillna(0).astype(int)
    df_medications_cardio = df_medications_cardio[['nsrrid', 'med_c']]
    df_medications_cardio.rename(columns={'nsrrid': 'id'}, inplace=True)
    df = pd.merge(df, df_medications_cardio, on='id', how='left')

    # Non-cardio medications
    medications_list_not_cardio = ["alzh", "apbs", "benzod", "hctz", "hctzk", "maoi", "ntca", "ostrd", "prknsn", "tca"]
    regex_pattern = f"^({'|'.join(medications_list_not_cardio)})(1|2)?$"
    matching_columns = [col for col in demographic_df_filtered.columns if re.match(regex_pattern, col)]
    df_medications_not_cardio = demographic_df_filtered[['nsrrid'] + matching_columns]
    # keep the max of the medications col list
    df_medications_not_cardio['med_nc'] = df_medications_not_cardio[matching_columns].max(axis=1).fillna(0).astype(int)
    df_medications_not_cardio = df_medications_not_cardio[['nsrrid', 'med_nc']]
    df_medications_not_cardio.rename(columns={'nsrrid': 'id'}, inplace=True)
    df = pd.merge(df, df_medications_not_cardio, on='id', how='left')

    return df

def dataframe_split_stratified(df_to_split: pd.DataFrame, config_csv: dict, test: Union[float, list], val: Union[float, list], random_state: int=42, save: bool=False, use_cache: bool=True, path_to_folder_reproducibility: str='', json_file_name: str=''):
    '''
    Split a dataframe into train, validation and test sets preserving the distribution of labels.
    '''

    # if os.path.exists(os.path.join(path_to_folder_reproducibility, json_file_name)) & use_cache:
    #     logger.info(f"Loading train, val and test from cache {os.path.join(path_to_folder_reproducibility, json_file_name)}")
    #     split = json.load(open(os.path.join(path_to_folder_reproducibility, json_file_name), 'r'))
    #     df_train = df_to_split[df_to_split['id'].isin(split['train'])]
    #     df_val = df_to_split[df_to_split['id'].isin(split['val'])]
    #     df_test = df_to_split[df_to_split['id'].isin(split['test'])]
    #     return df_train, df_val, df_test
    
    original_df = extract_csv(original_csv_file=config_csv['original_csv_file'], target_outcome='all', composite_outcome=False, imputing_method="censor_date", demographic_csv_file=config_csv['demographic_csv_file'], demographic_stratification_var=config_csv['demographic_stratification_var'])
    df = pd.merge(df_to_split, original_df, how='right')
    df_test = pd.DataFrame([])
    df_val = pd.DataFrame([])

    if isinstance(test, list):
        df_test = df_to_split[df_to_split['id'].isin(test)]
        df = df[~df['id'].isin(test)]
        test_size = 0
    else:
        test_size = test
    
    if isinstance(val, list):
        df_val = df_to_split[df_to_split['id'].isin(val)]
        df = df[~df['id'].isin(val)]
        val_size = 0
    else:
        val_size = val
    
    if not df.empty:
        df_train = df.copy()
        stratification_columns = [col for col in df_train.columns if col!= "id"]
        times_columns = [col for col in df_train.columns if "time" in col]
        y_train = df_train[stratification_columns]
        float_cols = y_train.select_dtypes(include=['float64']).columns.values
        binned_columns = [*times_columns, *float_cols]
        for continous_col in binned_columns:
            y_train[f'{continous_col}_bin'] = pd.qcut(y_train[continous_col].fillna(np.nanmedian(y_train[continous_col])), q=5, labels=False, duplicates='drop')
            y_train.drop(continous_col, axis=1, inplace=True)
        is_binary_split = y_train.shape[1] == 1 

        column_to_drop = [col for col in y_train.columns if sum(y_train[col].isna()) > 0]
        df_train.drop(column_to_drop, axis=1, inplace=True)
        y_train.drop(column_to_drop, axis=1, inplace=True)
    else:
        df_train = pd.DataFrame([])


    if test_size > 0 and len(df_train) > 0:
        if is_binary_split:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            ss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
        for train_index, test_index in ss.split(df_train, y_train):
            df_train, df_test = df_train.iloc[train_index], df_train.iloc[test_index]
            y_train, y_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    if val_size > 0 and len(df_train) > 0:
        if is_binary_split:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        else:
            ss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
        for train_index, val_index in ss.split(df_train, y_train):
            df_train, df_val = df_train.iloc[train_index], df_train.iloc[val_index]
            y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    if save:    
        if path_to_folder_reproducibility == "":
            warnings.warn("Path to csv file not specified")
            path_to_folder_reproducibility = "./reproducibility/"
        os.makedirs(path_to_folder_reproducibility, exist_ok=True)
        split = {'train': df_train['id'].tolist(), 'val': df_val['id'].tolist(), 'test': df_test['id'].tolist()}
        json.dump(split, open(os.path.join(path_to_folder_reproducibility, json_file_name), 'w'), indent=4)
    return df_to_split[df_to_split['id'].isin(df_train['id'])] if len(df_train) > 0 else pd.DataFrame([], columns=df_to_split.columns), df_to_split[df_to_split['id'].isin(df_val['id'])] if len(df_val) > 0 else pd.DataFrame([], columns=df_to_split.columns), df_to_split[df_to_split['id'].isin(df_test['id'])] if len(df_test) > 0 else pd.DataFrame([], columns=df_to_split.columns)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_best_threshold(model, dataloader, metric, task, average = 'macro'):
    """
    Find the best threshold that maximizes metric.
    Args:
        model (torch.nn.Module): Model to optimize threshold for
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data
        metric (str): metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        task (str): 'binary', 'multilabel'
        average (str): 'macro', 'micro', 'weighted'
    Returns:
        float: Best threshold
        float: Best metric score
    """
    model.eval()
    val_probs = []
    val_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            demo_features = None
            if len(batch) == 4:
                inputs, masks, labels, demo_features = batch
            else:
                inputs, masks, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            masks = {k: (v.to(device) if v is not None else None) for k, v in masks.items()}
            if demo_features is not None:
                demo_features = demo_features.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                logits = model(inputs, masks, demo_features)

            probs = torch.sigmoid(logits)
            val_probs.append(probs)
            val_labels.append(labels)

    y_probs = torch.cat(val_probs)
    y_true = torch.cat(val_labels)

    thresholds = np.arange(0.01, 1.00, 0.01)
    best_metric = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        match metric:
            case 'f1':
                score = F1Score(threshold = thresh, task = task, num_classes = y_true.shape[1], average=average).to(device)
            case 'precision':
                score = Precision(threshold = thresh, task = task, num_classes = y_true.shape[1], average=average).to(device)
            case 'recall':
                score = Recall(threshold = thresh, task = task, num_classes = y_true.shape[1], average=average).to(device)
            case 'accuracy':
                score = Accuracy(threshold = thresh, task = task, num_classes = y_true.shape[1], average=average).to(device)
        value_metric = score(y_probs, y_true) 
        
        if value_metric > best_metric:
            best_metric = value_metric
            best_thresh = thresh

    return best_thresh, best_metric

def compute_lengths(csv_path: str, data_folder: str, modality:str= "EEG", resolution_s: int = 1):

    n_exam = re.search(r'shhs(\d+)', os.path.basename(csv_path)).group(1)
    # Check if directory exists before listing
    modality_dir = os.path.join(data_folder, "features", modality, f"res_{resolution_s}s")
    if not os.path.exists(modality_dir):
        raise FileNotFoundError(f"Directory not found: {modality_dir}")
        
    channel = os.listdir(modality_dir)[0]
    lengths_data = []

    if isinstance(csv_path, str):
        df = pd.read_csv(csv_path)
    else:
        df = csv_path
        
    for id_pt in df['nsrrid']:
        data_path = os.path.join(data_folder, "features", modality.upper(), f"res_{resolution_s}s", channel.upper(), f"shhs{n_exam}-{int(id_pt)}.npy")
        if os.path.exists(data_path):
            signal = np.load(data_path)
            length = signal.shape[0]
            total_length_s = length*resolution_s
            lengths_data.append({'id': id_pt, 'length': total_length_s})
        
    lengths_df = pd.DataFrame(lengths_data)
    
    # Save to the same folder as the original CSV or a dedicated one
    if isinstance(csv_path, str):
        output_dir = os.path.dirname(csv_path)
        output_file = os.path.join(output_dir, Path(csv_path).stem + '_lengths.csv')
    else:
        output_file = "lengths.csv" # Fallback if dataframe passed directly
    
    lengths_df.to_csv(output_file, index=False)
    print(f"Lengths saved to {output_file}")

def compute_lengths_raw(csv_path: str, data_folder: str, modality:str='ECG'):
    n_exam = re.search(r'shhs(\d+)', os.path.basename(csv_path)).group(1)
    lengths_data = []
    df = pd.read_csv(csv_path)
    
    corrupt_cache_path = os.path.join(data_folder, "corrupted_files.txt")
    
    for id_pt in df['nsrrid']:
        data_path = os.path.join(data_folder, f"shhs{n_exam}-{int(id_pt)}.hdf5")
        if os.path.exists(data_path):
            try:
                with h5py.File(data_path, 'r', rdcc_nbytes=300 * 512 * 8 * 2) as hf:
                    signal = hf[modality.upper()]
                    length = signal.shape[0]
                    total_length_s = length/128
                    lengths_data.append({'id': id_pt, 'length': total_length_s})
            except Exception as e:
                print(f"Error reading {data_path}: {e}")
                try:
                    with open(corrupt_cache_path, "a") as f:
                        f.write(f"{id_pt}\n")
                except Exception:
                    pass
        
    lengths_df = pd.DataFrame(lengths_data)
    output_dir = os.path.dirname(csv_path)
    output_file = os.path.join(output_dir, Path(csv_path).stem + '_lengths_raw.csv')
    lengths_df.to_csv(output_file, index=False)
    print(f"Lengths saved to {output_file}")

def update_dict(base_params, update_params):
    for key, value in update_params.items():
        if isinstance(value, dict):
            base_params[key] = update_dict(base_params.get(key, {}), value)
        else:
            base_params[key] = value
    return base_params

def merge_sweep_config(base_config, sweep_config):
    """
    Updates base_config (nested dict) with values from sweep_config (WandB Config),
    handling dot-notation keys (e.g., 'model_config.mamba_config.d_model').
    """
    for key, value in sweep_config.items():
        if '.' in key:
            parts = key.split('.')
            target = base_config
            # Navigate to the parent dictionary
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            # Set the value
            target[parts[-1]] = value
    return base_config