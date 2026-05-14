"""
Raw signal dataset module.

Inherits from scripts.sources.dataset (MultimodalDataset, MultimodalDataModule)
and overrides only what is needed for raw HDF5-based signal loading:
  - __init__: skips resolution_input_embedding_s assertion, uses compute_lengths_raw
  - __getitem__: loads raw signals from HDF5 with dynamic channel matching
  - collate_fn: pads 2D [C, T] tensors (no feature dim); returns 4-tuple (no IDs)
  - pad_batch_raw: handles 2D tensors instead of 3D
  - RawMultimodalDataModule: instantiates RawMultimodalDataset instead of MultimodalDataset
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.sources.dataset import (
    MultimodalDataset as BaseMultimodalDataset,
    MultimodalDataModule as BaseMultimodalDataModule,
)
from scripts.sources.utils.utils import extract_csv, compute_lengths_raw

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Padding helper for raw 2D tensors [Channels, Time]
# (base pad_batch expects 3D [Channels, Time, Feature])
# ---------------------------------------------------------------------------
def pad_batch_raw(batch: list[torch.Tensor], padding_value=0.0):
    """Pad a list of [C, T] tensors to the same time length using pre-allocation."""
    lengths = [tensor.shape[1] for tensor in batch]
    max_length = max(lengths)
    batch_size = len(batch)
    C = batch[0].shape[0]

    padded_batch = torch.full((batch_size, C, max_length), padding_value, dtype=batch[0].dtype)
    for i, tensor in enumerate(batch):
        T = tensor.shape[1]
        padded_batch[i, :, :T] = tensor

    mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]
    mask = mask.long()

    return padded_batch, mask, torch.tensor(lengths)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RawMultimodalDataset(BaseMultimodalDataset):
    """
    Dataset for raw physiological signals stored in HDF5 files.

    Inherits label handling, demographic features, task detection,
    label_distribution, plot_label_distribution, and __len__ from
    BaseMultimodalDataset.  Only __init__, __getitem__, and collate_fn
    are overridden.
    """

    def __init__(self, config_csv: dict, data_folder: str, model_config: dict, dataframe: pd.DataFrame = None,
                 file_naming: str = 'shhs'):
        # --- replicate parent init WITHOUT the resolution assertions ---
        assert all(mod in ['eeg', 'eog', 'ecg'] for mod in model_config['modalities']), \
            "Unsupported modality found. Modalities must be among ['eeg', 'eog', 'ecg']."

        self.file_naming = file_naming

        if self.file_naming == 'shhs':
            match = re.search(r'shhs(\d+)', os.path.basename(config_csv['original_csv_file']))
            self.n_exam = match.group(1) if match else '1'
        else:
            self.n_exam = None

        if dataframe is None:
            if self.file_naming == 'shhs':
                self.df = extract_csv(**config_csv)
            else:
                # For non-SHHS CSVs (e.g. SleepFM), the CSV is already in the right format
                self.df = pd.read_csv(config_csv['original_csv_file'])
        else:
            self.df = dataframe

        self.data_folder = data_folder

        # FILTER CORRUPT FILES
        corrupt_cache_path = os.path.join(self.data_folder, "corrupted_files.txt")
        if os.path.exists(corrupt_cache_path):
            with open(corrupt_cache_path, 'r') as f:
                corrupt_ids = set([line.strip() for line in f if line.strip()])
            if corrupt_ids:
                initial_len = len(self.df)
                self.df = self.df[~self.df['id'].astype(str).isin(corrupt_ids)]
                if len(self.df) < initial_len:
                    logger.warning(f"Removed {initial_len - len(self.df)} corrupted files from dataset using cache.")

        self.config_modalities = model_config['modalities']
        self.sampling_rate_hz = model_config.get('sampling_rate_hz', {})
        self.max_signal_length_s = model_config.get('max_signal_length_s', None)  # truncate to N seconds

        # Demographic features (identical to parent)
        self.use_demographic_features = model_config.get('use_demographic_features', False)
        dem_var = config_csv.get('demographic_stratification_var', [])
        self.columns_demographic = [
            col for col in self.df.columns
            if any(keyword.lower() in col.lower() for keyword in dem_var)
        ]
        if self.use_demographic_features:
            if dem_var == []:
                warnings.warn("Demographic features specified but no demographic variable found in the configuration file.")
            self.demo_data = self.df[self.columns_demographic].copy()

        # Labels (identical to parent)
        self.labels = self.df[[
            col for col in self.df.columns
            if 'time' not in col and col != 'id'
            and not any(keyword.lower() in col.lower() for keyword in dem_var)
        ]]
        self.labels_names = self.labels.columns.tolist()

        # Lengths — use raw-signal version (only for SHHS; SleepFM files are too large for pre-scanning)
        if self.file_naming == 'shhs':
            lengths_file = os.path.join(
                Path(config_csv['original_csv_file']).parent,
                Path(config_csv['original_csv_file']).stem + '_lengths_raw.csv',
            )
            if not os.path.exists(lengths_file):
                compute_lengths_raw(config_csv['original_csv_file'], data_folder)

            lengths_df = pd.read_csv(lengths_file)
            lengths_map = dict(zip(lengths_df['id'], lengths_df['length']))
            self.lengths = [lengths_map.get(uid, 0) for uid in self.df['id']] if not self.df.empty else None
        else:
            # For non-SHHS datasets, lengths will be determined dynamically at load time
            self.lengths = None

        # Task type (identical to parent)
        self.task = "classification" if all("time" not in col.lower() for col in self.df.columns) else "survival"
        self.n_labels = self.labels.shape[1]
        self.task += "_binary" if self.n_labels == 1 else "_multilabel"

        if "survival" in self.task:
            if self.n_labels == 1:
                self.times = self.df[['time']]
                self.times_names = ['time']
            else:
                self.times = self.df[[col + "_time" for col in self.labels.columns]]
                self.times_names = self.times.columns.tolist()

    # ----- label_distribution, plot_label_distribution, __len__ inherited -----

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        multimodal_data = {}

        # Build HDF5 path based on naming convention
        if self.file_naming == 'sleepfm':
            # SleepFM: Study ID IS the filename (e.g. SSC_2004_1083224360.hdf5)
            data_path = os.path.join(self.data_folder, f"{row['id']}.hdf5")
        else:
            # SHHS: shhs{n_exam}-{int_id}.hdf5
            data_path = os.path.join(self.data_folder, f"shhs{self.n_exam}-{int(row['id'])}.hdf5")

        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} not found.")

        try:
            # We don't allocate the massive chunk cache (rdcc_nbytes) since we process single segments.
            # This drastically reduces I/O overhead on concurrent loader threads.
            with h5py.File(data_path, 'r') as hf:
                available_keys = list(hf.keys())
                used_keys = set()

                # Track channel mapping for logging
                channel_mapping = {}

                for modality, channels in self.config_modalities.items():
                    modality_data = []
                    channel_mapping[modality] = []

                    for channel in channels:
                        matched_key = None
                        for k in available_keys:
                            if channel.lower() in k.lower() and k not in used_keys:
                                matched_key = k
                                used_keys.add(k)
                                channel_mapping[modality].append(f"{channel} -> {matched_key}")
                                break

                        if matched_key is None:
                            raise ValueError(
                                f"Could not find an unused channel matching '{channel}' "
                                f"in {data_path}. Available: {available_keys}"
                            )

                        raw_data = hf[matched_key][:]
                        tensor = torch.tensor(raw_data, dtype=torch.float32).squeeze()

                        # Truncate to max_signal_length_s if specified
                        if self.max_signal_length_s is not None:
                            fs = self.sampling_rate_hz.get(modality, 128)
                            max_samples = int(fs * self.max_signal_length_s)
                            if tensor.shape[-1] > max_samples:
                                tensor = tensor[..., :max_samples]

                        modality_data.append(tensor)

                    if modality_data:
                        multimodal_data[modality] = torch.stack(modality_data, dim=0)

                # Log channel mapping
                if idx == 0:
                    logger.info(f"Sample channel mapping (Patient {row['id']}): {channel_mapping}")
                else:
                    logger.debug(f"Channel mapping (Patient {row['id']}): {channel_mapping}")

        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            corrupt_cache_path = os.path.join(self.data_folder, "corrupted_files.txt")
            try:
                with open(corrupt_cache_path, "a") as f:
                    f.write(f"{row['id']}\n")
            except Exception as cache_e:
                logger.error(f"Failed to write to corrupt files cache: {cache_e}")
            
            # Fetch a random different index to prevent training crash
            import random
            if len(self.df) > 1:
                new_idx = random.randint(0, len(self.df) - 1)
                while new_idx == idx:
                    new_idx = random.randint(0, len(self.df) - 1)
                logger.warning(f"Skipping corrupted file for id {row['id']} and sampling index {new_idx} instead.")
                return self.__getitem__(new_idx)
            else:
                raise RuntimeError(f"Corrupted file: {data_path}, and no other files available to sample.") from e

        # Labels & survival times (identical to parent)
        labels = torch.tensor(self.labels.iloc[idx, :].values, dtype=torch.float32)
        if "survival" in self.task:
            times = torch.tensor(self.times.iloc[idx, :].values, dtype=torch.float32)
            labels = torch.stack((labels, times), dim=1)

        demo_tensor = torch.empty(0)
        if self.use_demographic_features:
            demo_values = self.demo_data.iloc[idx].values.astype(float)
            demo_tensor = torch.tensor(demo_values, dtype=torch.float32)

        # Return ID — for SHHS it's an int, for SleepFM it's a string
        # Kept as a plain Python value (not a tensor) so string IDs work
        try:
            patient_id = int(row['id'])
        except (ValueError, TypeError):
            patient_id = str(row['id'])

        return multimodal_data, labels, demo_tensor, patient_id

    def collate_fn(self, batch, padding_value=0.0):
        """Collate raw signal batches. Returns 5-tuple (inputs, masks, labels, demo_features, ids)."""
        modalities = list(batch[0][0].keys())

        inputs = {}
        masks = {}

        labels = torch.stack([sample[1] for sample in batch], dim=0)
        if "survival" not in self.task:
            labels = labels.to(torch.int64)

        ids = [sample[3] for sample in batch]  # plain Python list (int or str)

        demo_features = None
        if len(batch[0]) > 3 and batch[0][2].numel() > 0:
            demo_features = torch.stack([sample[2] for sample in batch], dim=0)

        for modality in modalities:
            tensor_list = [sample[0][modality] for sample in batch]
            padded_tensors, mask, lengths = pad_batch_raw(tensor_list, padding_value=padding_value)
            inputs[modality] = padded_tensors   # [Batch, Channel, Seq_len]
            masks[modality] = mask              # [Batch, Seq_len]

        return inputs, masks, labels, demo_features, ids


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class RawMultimodalDataModule(BaseMultimodalDataModule):
    """
    Lightning DataModule for raw signals.

    Replicates the parent's splitting/scaling logic but instantiates
    RawMultimodalDataset instead of MultimodalDataset.
    
    Inherits train_dataloader, val_dataloader, test_dataloader from parent.
    """

    def __init__(self, config_csv: dict, data_folder: str, model_config: dict,
                 batch_size: int = 32, num_workers: int = 4,
                 scaler=None, use_cache: bool = True,
                 random_state: int = 42,
                 test: Union[float, list] = 0.2,
                 val: Union[float, list] = 0.2,
                 save_splits: bool = False,
                 file_naming: str = 'shhs'):
        # Skip BaseMultimodalDataModule.__init__ — go straight to LightningDataModule
        import pytorch_lightning as pl
        pl.LightningDataModule.__init__(self)

        self.config_csv = config_csv
        self.data_folder = data_folder
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test = test
        self.val = val
        self.random_state = random_state
        self.save_splits = save_splits
        self.use_cache = use_cache
        self.scaler = scaler
        self.file_naming = file_naming

        # ------ Dataset splitting (same logic as parent) ------
        from scripts.sources.utils.utils import dataframe_split_stratified
        import pickle as pkl
        from sklearn.preprocessing import StandardScaler
        from pathlib import Path

        if self.file_naming == 'shhs':
            self.full_dataset = extract_csv(**self.config_csv)
        else:
            # For non-SHHS CSVs, read directly (already properly formatted)
            self.full_dataset = pd.read_csv(self.config_csv['original_csv_file'])

        # FILTER CORRUPT FILES
        corrupt_cache_path = os.path.join(self.data_folder, "corrupted_files.txt")
        if os.path.exists(corrupt_cache_path):
            with open(corrupt_cache_path, 'r') as f:
                corrupt_ids = set([line.strip() for line in f if line.strip()])
            if corrupt_ids:
                initial_len = len(self.full_dataset)
                self.full_dataset = self.full_dataset[~self.full_dataset['id'].astype(str).isin(corrupt_ids)]
                if len(self.full_dataset) < initial_len:
                    logger.warning(f"Removed {initial_len - len(self.full_dataset)} corrupted files from DataModule using cache.")

        self.dem_var = self.config_csv.get('demographic_stratification_var', [])
        self.demographic_columns = [
            col for col in self.full_dataset.columns
            if any(kw.lower() in col.lower() for kw in self.dem_var)
        ]
        self.labels_names = [
            col for col in self.full_dataset.columns
            if 'time' not in col and col != 'id' and col not in self.demographic_columns
        ]
        self.task = "survival" if any("time" in col.lower() for col in self.full_dataset.columns) else "classification"
        self.n_labels = len(self.labels_names)
        self.task += "_binary" if self.n_labels == 1 else "_multilabel"

        # Robust dataset_events_name extraction (handles both SHHS and SleepFM CSV names)
        csv_stem = Path(self.config_csv['original_csv_file']).stem
        if 'from_' in csv_stem:
            self.dataset_events_name = csv_stem.split('from_')[1].split('_')[0]
        else:
            self.dataset_events_name = csv_stem

        self.reproducibility_path = (
            os.path.join(Path(self.config_csv['original_csv_file']).parent,
                         'reproducibility', self.dataset_events_name, '_'.join(self.dem_var))
            if 'reproducibility' not in str(Path(self.config_csv['original_csv_file']).parent)
            else str(Path(self.config_csv['original_csv_file']).parent)
        )
        self.json_splits_file_name = (
            Path(self.config_csv['original_csv_file']).stem
            + ('_th' + str(self.config_csv['threshold_time']) if 'classification' in self.task else '')
            + '_split.json'
        )

        if self.file_naming == 'shhs':
            self.train_dataframe, self.val_dataframe, self.test_dataframe = dataframe_split_stratified(
                df_to_split=self.full_dataset,
                config_csv=self.config_csv,
                test=self.test,
                val=self.val,
                random_state=self.random_state,
                save=self.save_splits,
                use_cache=self.use_cache,
                path_to_folder_reproducibility=self.reproducibility_path,
                json_file_name=self.json_splits_file_name,
            )
        else:
            # For non-SHHS datasets, do a direct stratified split
            # (dataframe_split_stratified calls extract_csv internally which is SHHS-specific)
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            from sklearn.model_selection import StratifiedShuffleSplit as SklearnStratifiedSplit

            df = self.full_dataset.copy()
            label_cols = self.labels_names
            y = df[label_cols]
            is_binary = len(label_cols) == 1

            # Split for test
            if isinstance(self.test, list):
                test_df = df[df['id'].isin(self.test)]
                df = df[~df['id'].isin(self.test)]
                y = df[label_cols]
            elif self.test > 0:
                if is_binary:
                    ss = SklearnStratifiedSplit(n_splits=1, test_size=self.test, random_state=self.random_state)
                else:
                    ss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.test, random_state=self.random_state)
                for train_idx, test_idx in ss.split(df, y):
                    test_df = df.iloc[test_idx]
                    df = df.iloc[train_idx]
                    y = df[label_cols]
            else:
                test_df = pd.DataFrame(columns=df.columns)

            # Split for val
            if isinstance(self.val, list):
                val_df = df[df['id'].isin(self.val)]
                df = df[~df['id'].isin(self.val)]
            elif self.val > 0:
                if is_binary:
                    ss = SklearnStratifiedSplit(n_splits=1, test_size=self.val, random_state=self.random_state)
                else:
                    ss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.val, random_state=self.random_state)
                y = df[label_cols]
                for train_idx, val_idx in ss.split(df, y):
                    val_df = df.iloc[val_idx]
                    df = df.iloc[train_idx]
            else:
                val_df = pd.DataFrame(columns=df.columns)

            self.train_dataframe = df.reset_index(drop=True)
            self.val_dataframe = val_df.reset_index(drop=True)
            self.test_dataframe = test_df.reset_index(drop=True)

            logger.info(f"Split: train={len(self.train_dataframe)}, val={len(self.val_dataframe)}, test={len(self.test_dataframe)}")

            # Save splits
            if self.save_splits:
                os.makedirs(self.reproducibility_path, exist_ok=True)
                split_dict = {
                    'train': self.train_dataframe['id'].tolist(),
                    'val': self.val_dataframe['id'].tolist(),
                    'test': self.test_dataframe['id'].tolist(),
                }
                split_path = os.path.join(self.reproducibility_path, self.json_splits_file_name)
                with open(split_path, 'w') as f:
                    json.dump(split_dict, f, indent=4)
                logger.info(f"Splits saved to: {split_path}")

        # ------ Demographic scaling (same logic as parent) ------
        if self.demographic_columns:
            self.scaled_columns = []
            for col in self.demographic_columns:
                if 'gender' in col:
                    for df in [self.train_dataframe, self.val_dataframe, self.test_dataframe]:
                        if not df.empty and col in df:
                            df.loc[:, col] = df.loc[:, col] - 1
                elif (self.train_dataframe.loc[:, col].value_counts().shape[0] > 2
                      or self.test_dataframe.loc[:, col].value_counts().shape[0] > 2):
                    for df in [self.train_dataframe, self.val_dataframe, self.test_dataframe]:
                        if not df.empty and col in df:
                            df[col] = df[col].astype(float)
                    self.scaled_columns.append(col)

            scaler_path = os.path.join(
                self.reproducibility_path,
                ('th' + str(self.config_csv.get('threshold_time', '')) + '_' if 'classification' in self.task else 'surv_')
                + 'demographics_scaler.pkl',
            ) if "balanced" not in self.json_splits_file_name else os.path.join(
                self.reproducibility_path,
                ('th' + str(self.config_csv.get('threshold_time', '')) + '_' if 'classification' in self.task else 'surv_')
                + f'demographics_scaler_{self.json_splits_file_name.split("age_")[-1].split("_split")[0]}.pkl',
            )

            if scaler is not None:
                self.scaler = scaler
            elif os.path.exists(scaler_path):
                logger.info(f"Loading demographics scaler from: {scaler_path}")
                self.scaler = pkl.load(open(scaler_path, 'rb'))
            else:
                self.scaler = StandardScaler()
                if self.scaled_columns != []:
                    logger.info(f"Fitting new demographics scaler with {self.scaled_columns}")
                    self.scaler.fit(self.train_dataframe.loc[:, self.scaled_columns])
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    with open(scaler_path, 'wb') as f:
                        pkl.dump(self.scaler, f)

            if hasattr(self.scaler, 'feature_names_in_'):
                self.scaled_columns = list(self.scaler.feature_names_in_)

            for df in [self.train_dataframe, self.val_dataframe, self.test_dataframe]:
                if not df.empty and self.scaled_columns != []:
                    df.loc[:, self.scaled_columns] = self.scaler.transform(df.loc[:, self.scaled_columns])

        # ------ Create RawMultimodalDataset instances ------
        columns_to_drop = []

        self.train_dataset = RawMultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.train_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )
        self.val_dataset = RawMultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.val_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )
        self.test_dataset = RawMultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.test_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )

# ---------------------------------------------------------------------------
# Stratified Variants
# ---------------------------------------------------------------------------
class StratifiedRawMultimodalDataset(RawMultimodalDataset):
    """
    Extends RawMultimodalDataset to emit an integer age 'stratum_id' per subject
    based on the provided thresholds.
    """
    def __init__(self, n_bins: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        # Find exactly the column with 'age' in its name
        self.age_col = None
        for col in self.df.columns:
            if 'age' in col.lower() and 'time' not in col.lower():
                self.age_col = col
                break

        if self.age_col is None:
            raise ValueError(
                "Could not find an age column in DataFrame to stratify on. "
                f"Available cols: {self.df.columns.tolist()}"
            )
            
        ages = self.df[self.age_col].values
        self.stratum_ids = pd.qcut(ages, q=self.n_bins, labels=False, duplicates='drop')

    def __getitem__(self, idx):
        multimodal_data, labels, demo_tensor, _id = super().__getitem__(idx)
        stratum_id = torch.tensor(self.stratum_ids[idx], dtype=torch.long)
        return multimodal_data, labels, demo_tensor, stratum_id, _id

    def collate_fn(self, batch, padding_value=0.0):
        # batch element: (multimodal_data, labels, demo_tensor, stratum_id, id)
        # Drop stratum_id & id to reuse parent collater, then append them back
        parent_batch = [(s[0], s[1], s[2], s[4]) for s in batch]
        inputs, masks, labels, demo_features, ids = super().collate_fn(parent_batch, padding_value)
        
        stratum_ids = torch.stack([s[3] for s in batch], dim=0)
        return inputs, masks, labels, demo_features, stratum_ids, ids

class StratifiedRawMultimodalDataModule(RawMultimodalDataModule):
    def __init__(self, n_bins: int = 10, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)

    def _init_datasets(self):
        # We override super methods to initialize the stratified dataset instead
        pass 

    def __init__(self, n_bins: int = 10, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)
        # Re-assign self.train_dataset, self.val_dataset, self.test_dataset
        
        columns_to_drop = []
        
        self.train_dataset = StratifiedRawMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.train_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )
        self.val_dataset = StratifiedRawMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.val_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )
        self.test_dataset = StratifiedRawMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.test_dataframe.drop(columns=columns_to_drop, axis=1),
            file_naming=self.file_naming,
        )