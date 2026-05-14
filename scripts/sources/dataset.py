import pandas as pd
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from pyedflib import highlevel
from torch.nn.utils.rnn import pad_sequence
from typing import Union
import os
import json
import re
import logging
from pathlib import Path
from typing import Optional
from einops import rearrange
from scripts.sources.utils.utils import extract_csv, compute_lengths, dataframe_split_stratified
from scripts.sources.utils.samplers import SmartBatchSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def pad_batch(batch: list[torch.Tensor], padding_value=0.0):
    # Determine lengths along the time dimension (assumed at dim=1)
    lengths = [tensor.shape[1] for tensor in batch]
    max_length = max(lengths)

    padded_list = []
    for tensor in batch:
        # [Channels, Time, Feature] case of embeddings
        C, T, F = tensor.shape
        if T < max_length:
            pad_shape = (C, max_length - T, F)
            pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype)
            padded = torch.cat([tensor, pad], dim=1)
        else:
            padded = tensor

        padded_list.append(padded)
        
    mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]
    mask = mask.long()
    
    return torch.stack(padded_list, dim=0), mask, torch.tensor(lengths)

class MultimodalDataset(Dataset):
    def __init__(self, config_csv: dict, data_folder: str, model_config: dict, dataframe: pd.DataFrame = None):
        """
        Inputs:
            config_csv (dict): Dictionary containing the configuration for hadle the csv file.
            data_folder (str): Directory with all the data files.
            model_config (dict): Dictionary containing the configuration for the model. Specifically, for Dataset construction we need:
                - modalities (dict): Dictionary specifying which modalities to include and the corresponding channels
                - resolution_input_embedding_s (dict): Dictionary specifying the embedding resolution for each modality (in s: the length of the signal which the token correspond to)
                - resolution_output_embedding_s (int): Embedding resolution for the outcome (in s: the length of the signal which the token correspond to)
                - use_demographic_features (bool, optional): Whether to load demographic features
            dataframe (pd.DataFrame, optional): DataFrame containing the dataset. If None, the dataset will be calculated from the csv file.
        Output:
            inputs: {
                'eeg': [Channels, Time] or [Channels, Time, Feature],
                'eog': [Channels, Time] or [Channels, Time, Feature],
                'ecg': [Channels, Time] or [Channels, Time, Feature],
                }
            labels: [Num_Labels], # [Num_Labels, 2] for survival task (event, time)
            demo_features: [Num_Demo_Features],
        Collate function:
            inputs: {
                'eeg': [Batch, Channels, Time] or [Batch, Channels, Time, Feature],
                'eog': [Batch, Channels, Time] or [Batch, Channels, Time, Feature],
                'ecg': [Batch, Channels, Time] or [Batch, Channels, Time, Feature],
                }
            mask: [Batch, Time]
            labels: [Batch, Num_Labels], # [Batch, Num_Labels, 2] for survival task (event, time)
            demo_features: [Batch, Num_Demo_Features],
        """
        
        assert all(mod in ['eeg', 'eog', 'ecg'] for mod in model_config['modalities'].keys()), "Unsupported modality found. Modalities must be among ['EEG', 'EOG', 'ECG']."
        assert model_config['modalities'].keys() == model_config['resolution_input_embedding_s'].keys(), "Modalities in config_modalities and resolution_input_embedding_s must be the same."

        self.n_exam = re.search(r'shhs(\d+)', os.path.basename(config_csv['original_csv_file'])).group(1)

        if dataframe is None:
            self.df = extract_csv(**config_csv)
        else:
            self.df = dataframe
        # if len(self.df)==0:
        #     raise ValueError("Empty DataFrame. Check the configuration file.")
        self.data_folder = data_folder
        self.config_modalities = model_config['modalities']
        self.resolution_input_embedding_s = model_config['resolution_input_embedding_s']
        self.resolution_output_embedding_s = model_config['resolution_output_embedding_s']
        
        # Demographic features handling
        self.use_demographic_features = model_config.get('use_demographic_features', False)
        dem_var = config_csv.get('demographic_stratification_var', [])
        self.columns_demographic = [col for col in self.df.columns if any(keyword.lower() in col.lower() for keyword in dem_var)]
        if self.use_demographic_features:
            if dem_var==[]:
                warnings.warn("Demographic features specified but no demographic variable found in the configuration file.")
            self.demo_data = self.df[self.columns_demographic].copy()
        
        self.labels = self.df[[col for col in self.df.columns if 'time' not in col and col!='id' and not any(keyword.lower() in col.lower() for keyword in dem_var)]]
        self.labels_names = self.labels.columns.tolist()

        # Load lengths if available
        lengths_file = os.path.join(Path(config_csv['original_csv_file']).parent, Path(config_csv['original_csv_file']).stem + '_lengths.csv')
        if not os.path.exists(lengths_file):
            compute_lengths(config_csv['original_csv_file'], data_folder, list(self.resolution_input_embedding_s.keys())[0].upper(), list(self.resolution_input_embedding_s.values())[0])

        lengths_df = pd.read_csv(lengths_file)
        # Map id to length. Ensure alignment with self.df
        # self.df has an 'id' column.
        # We create a length array aligned with self.df
        lengths_map = dict(zip(lengths_df['id'], lengths_df['length']))
        self.lengths = [lengths_map.get(uid, 0) for uid in self.df['id']] if not self.df.empty else None
            
        self.task = "classification" if all("time" not in col.lower() for col in self.df.columns) else "survival"
        self.n_labels = self.labels.shape[1]
        if self.n_labels==1:
            self.task += "_binary"
        else:
            self.task += "_multilabel"

        if "survival" in self.task:
            if self.n_labels == 1:
                self.times = self.df[['time']]
                self.times_names = ['time']
            else:
                self.times = self.df[[col+"_time" for col in self.labels.columns]]
                self.times_names = self.times.columns.tolist()
    
    def label_distribution(self):
        distribution = {}
        for col in self.labels.columns:
            distribution[col] = self.labels[col].value_counts().to_dict()
        return distribution
    
    def plot_label_distribution(self, subtitle: Optional[str] = None, save_dir: Optional[str] = None):
        if len(self.labels.columns) > 1:
            plt.figure(figsize=(12, 6))
            counts = (self.labels == 1).sum()
            plt.bar(counts.index, [self.__len__()] * len(counts), color='lightgray', alpha=0.5, label=f'Total Samples ({self.__len__()} samples)')
            bars = plt.bar(counts.index, counts.values, color='tab:blue', label='Events')
            plt.bar_label(bars, padding=3)
            plt.title(f"Label Distribution - {subtitle}")
            plt.ylabel("Counts")
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{subtitle.lower() + '_' if subtitle else ''}full_labels_distribution.png"
                plt.savefig(os.path.join(save_dir, filename))
                plt.close()
            else:
                plt.show()
        else:
            col = self.labels.columns[0]
            plt.figure(figsize=(10, 6))
            counts = (self.labels[col] == 1).sum()
            plt.bar(col, self.__len__(), color='lightgray', alpha=0.5, label=f'Total Samples ({self.__len__()})', width=0.5)
            bars = plt.bar(col, counts, color='tab:blue', label='Events', width=0.5)
            plt.bar_label(bars, padding=3)
            plt.title(f"Label Distribution - {subtitle}")
            plt.ylabel("Counts")
            plt.xticks(rotation=45, ha='right')
            plt.xlim(-1,1)
            plt.legend()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{subtitle.lower() + '_' if subtitle else ''}{col.lower()}_distribution.png"
                plt.savefig(os.path.join(save_dir, filename))
                plt.close()
            else:
                plt.show()
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]

        multimodal_data = {}

        tensor = None

        for modality, channels in self.config_modalities.items():

            modality_data = []

            for channel in channels:
                tensor = None
                res = f"res_{self.resolution_input_embedding_s[modality]}s" if self.resolution_input_embedding_s[modality] is not None else ""
                data_path = os.path.join(self.data_folder, "features", modality.upper(), res, channel.upper(), f"shhs{self.n_exam}-{int(row['id'])}.npy")

                if os.path.exists(data_path):
                    pre_processed_embedding = np.load(data_path) # different resolution for eeg and eog
                    # dropping last elements to have a signal length multiple of factor resolution
                    factor_resolution = int(self.resolution_output_embedding_s / self.resolution_input_embedding_s[modality])
                    pre_processed_embedding = pre_processed_embedding[:len(pre_processed_embedding) - len(pre_processed_embedding) % factor_resolution, :]
                else:
                    raise ValueError(f"Data path {data_path} not found.")
                tensor = torch.tensor(pre_processed_embedding, dtype=torch.float32).squeeze() # [Seq_len_input_resolution, Feature_dim]
                modality_data.append(tensor)
            if modality_data:
                # Shape embedding: [Num_Canali, Seq_len_input_resolution, Feature_dim]
                multimodal_data[modality] = torch.stack(modality_data, dim=0)
        
        labels = torch.tensor(self.labels.iloc[idx,:].values, dtype=torch.float32)

        if "survival" in self.task:
            times = torch.tensor(self.times.iloc[idx,:].values, dtype=torch.float32)
            labels = torch.stack((labels, times), dim=1)

        demo_tensor = torch.empty(0)
        if self.use_demographic_features:
             # Extract demographic features for this index
             # Ensure they are numeric
             demo_values = self.demo_data.iloc[idx].values.astype(float)
             demo_tensor = torch.tensor(demo_values, dtype=torch.float32)


        return multimodal_data, labels, demo_tensor, torch.tensor(int(row['id']))

    def collate_fn(self, batch, padding_value=0.0):
        modalities = [key for key in batch[0][0].keys()]

        inputs = {}
        masks = {}

        labels = [sample[1] for sample in batch]
        labels = torch.stack(labels, dim=0) # [Batch, Num_Labels]
        if "survival" not in self.task:
            labels = labels.to(torch.int64)
        
        ids = [sample[-1] for sample in batch]
        ids = torch.stack(ids, dim=0)

        demo_features = None
        if len(batch[0]) > 3 and batch[0][2].numel() > 0:
            demo_features = [sample[2] for sample in batch]
            demo_features = torch.stack(demo_features, dim=0) # [Batch, Num_Demo_Features]

        for modality in modalities:
            tensor_list = [sample[0][modality] for sample in batch]

            padded_tensors, mask, lengths = pad_batch(tensor_list, padding_value=padding_value)
            inputs[modality] = padded_tensors # [Batch, Channel, Seq_len_input_resolution_mod, Feature_dim]
            masks[modality] = mask # [Batch, Seq_len_input_resolution_mod] I need the modality specific mask because of the channel pooling possible using attention
        
        return inputs, masks, labels, demo_features, ids

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, config_csv: dict, data_folder: str, model_config: dict, batch_size: int = 32, num_workers: int = 4, scaler: Optional[StandardScaler] = None, use_cache: bool = True, random_state: int = 42, test: Union[float, list] = 0.2, val: Union[float, list] = 0.2, save_splits: bool = False):
        """
        Args:
            config_csv (dict): Configuration for the CSV file.
            data_folder (str): Path to the data folder.
            model_config (dict): Configuration for the model.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            scaler (Optional[StandardScaler], optional): Scaler to use for the data. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to True.
            random_state (int, optional): Random state. Defaults to 42.
            test (Union[float, list], optional): Either test size or the list of indices for test. Default to 0.2
            val (Union[float, list], optional): Validation size or list of indices for validation. Defaults to 0.2.
            save_splits (bool, optional): Whether to save the splits. Defaults to False.
        """
        super().__init__()
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

        # Create full dataset
        self.full_dataset = extract_csv(**self.config_csv)
        self.dem_var = self.config_csv.get('demographic_stratification_var', [])
        self.demographic_columns = [col for col in self.full_dataset.columns if any(keyword.lower() in col.lower() for keyword in self.dem_var)]
        columns_to_drop = [] #[col for col in self.full_dataset.columns if any(keyword.lower() in col.lower() for keyword in self.dem_var)]

        self.labels_names = [col for col in self.full_dataset.columns if 'time' not in col and col!='id' and col!='time' and col not in self.demographic_columns]
        self.task = "survival" if any("time" in col.lower() for col in self.full_dataset.columns) else "classification"
        self.n_labels = len(self.labels_names)
        if self.n_labels==1:
            self.task += "_binary"
        else:
            self.task += "_multilabel"

        self.dataset_events_name = Path(self.config_csv['original_csv_file']).stem.split('from_')[1].split('_')[0]
        
        self.reproducibility_path = os.path.join(Path(self.config_csv['original_csv_file']).parent, 'reproducibility', self.dataset_events_name, '_'.join(self.dem_var)) if not 'reproducibility' in str(Path(self.config_csv['original_csv_file']).parent) else Path(self.config_csv['original_csv_file']).parent
        self.json_splits_file_name = Path(self.config_csv['original_csv_file']).stem + ['_th' + str(self.config_csv['threshold_time']) if 'classification' in self.task else ''][0] + '_split.json'

        self.train_dataframe, self.val_dataframe, self.test_dataframe = dataframe_split_stratified(
                df_to_split=self.full_dataset,
                config_csv=self.config_csv,
                test=self.test,
                val=self.val,
                random_state=self.random_state,
                save=self.save_splits,
                use_cache=self.use_cache,
                path_to_folder_reproducibility=self.reproducibility_path,
                json_file_name=self.json_splits_file_name
            )
        if self.demographic_columns!=[]:
            self.scaled_columns = []
            # Explicitly cast to float to avoid Int64 incompatibility
            # Iterating to ensure dtypes are updated correctly
            for col in self.demographic_columns:
                if 'gender' in col:
                    if not self.train_dataframe.empty and col in self.train_dataframe:
                        self.train_dataframe.loc[:, col] = self.train_dataframe.loc[:, col]-1
                    if not self.val_dataframe.empty and col in self.val_dataframe:
                        self.val_dataframe.loc[:, col] = self.val_dataframe.loc[:, col]-1
                    if not self.test_dataframe.empty and col in self.test_dataframe:
                        self.test_dataframe.loc[:, col] = self.test_dataframe.loc[:, col]-1
                elif self.train_dataframe.loc[:, col].value_counts().shape[0]>2 or self.test_dataframe.loc[:, col].value_counts().shape[0]>2: # binary label
                    if not self.train_dataframe.empty and col in self.train_dataframe:
                        self.train_dataframe[col] = self.train_dataframe[col].astype(float)
                    if not self.val_dataframe.empty and col in self.val_dataframe:
                        self.val_dataframe[col] = self.val_dataframe[col].astype(float)
                    if not self.test_dataframe.empty and col in self.test_dataframe:
                        self.test_dataframe[col] = self.test_dataframe[col].astype(float)
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
                logger.info(f"Loading demographics scaler from file: {scaler_path}")
                self.scaler = pkl.load(open(scaler_path, 'rb'))
                logger.info(f'Means: {self.scaler.mean_}')
                logger.info(f'Vars: {self.scaler.var_}')
                logger.info(f'Samples: {self.scaler.n_samples_seen_}')
            else:
                if self.scaled_columns != []:
                    self.scaler = StandardScaler()
                    logger.info(f"Fitting new demographics scaler with {self.scaled_columns}")
                    self.scaler.fit(self.train_dataframe.loc[:, self.scaled_columns])
                    os.makedirs(self.reproducibility_path, exist_ok=True)
                    with open(scaler_path, 'wb') as f:
                        pkl.dump(self.scaler, f)
                    logger.info(f'Means: {self.scaler.mean_}')
                    logger.info(f'Vars: {self.scaler.var_}')
                    logger.info(f'Samples: {self.scaler.n_samples_seen_}')

            if hasattr(self.scaler, 'feature_names_in_'):
                assert self.scaled_columns == list(self.scaler.feature_names_in_), f"Scaled columns {self.scaled_columns} do not match feature names in scaler {list(self.scaler.feature_names_in_)}"
                self.scaled_columns = list(self.scaler.feature_names_in_)

            self.train_dataframe.loc[:, self.scaled_columns] = self.scaler.transform(self.train_dataframe.loc[:, self.scaled_columns]) if (not self.train_dataframe.empty and self.scaled_columns != []) else None
            self.val_dataframe.loc[:, self.scaled_columns] = self.scaler.transform(self.val_dataframe.loc[:, self.scaled_columns]) if (not self.val_dataframe.empty and self.scaled_columns != []) else None
            self.test_dataframe.loc[:, self.scaled_columns] = self.scaler.transform(self.test_dataframe.loc[:, self.scaled_columns]) if (not self.test_dataframe.empty and self.scaled_columns != []) else None

        self.train_dataset = MultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.train_dataframe.drop(columns=columns_to_drop, axis=1),
        )

        self.val_dataset = MultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.val_dataframe.drop(columns=columns_to_drop, axis=1),
        )
    
        self.test_dataset = MultimodalDataset(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.test_dataframe.drop(columns=columns_to_drop, axis=1),
        )
    
    def __len__(self):
        return len(self.full_dataset)
    
    def __getitem__(self, idx):
        full_dataset = MultimodalDataset(
            config_csv = self.config_csv,
            data_folder = self.data_folder,
            model_config = self.model_config,
            dataframe=self.full_dataset
        )
        return full_dataset[idx]
    
    def label_distribution(self, type_dataset: Optional[str] = None):
        if type_dataset is None:
            outcomes = {}
            for col in self.labels_names:
                outcomes[col] = self.full_dataset[col].value_counts().to_dict()
            return outcomes
        elif type_dataset == "train":
            return self.train_dataset.label_distribution()
        elif type_dataset == "val":
            return self.val_dataset.label_distribution()
        elif type_dataset == "test":
            return self.test_dataset.label_distribution()
        else:
            raise ValueError("type_dataset must be 'train', 'val', 'test' or None")
    
    def plot_label_distribution(self, type_dataset: Optional[str] = None, save_dir: Optional[str] = None):
        if type_dataset is None:
            total_len = len(self.full_dataset)
            if len(self.labels_names) > 1:
                plt.figure(figsize=(12, 6))
                counts = (self.full_dataset[self.labels_names] == 1).sum()
                plt.bar(counts.index, total_len, color='lightgray', alpha=0.5, label=f'Total Samples ({total_len})')
                bars = plt.bar(counts.index, counts.values, color='tab:blue', label='Events')
                plt.bar_label(bars, padding=3)
                # plt.bar(counts.index, counts.values, color='tab:blue', label='Events')
                plt.title("Label Distribution")
                plt.ylabel("Counts")
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, "full_labels_distribution.png"))
                    plt.close()
                else:
                    plt.show()
            else:
                col = self.labels_names[0]
                plt.figure(figsize=(10, 6))
                counts = self.full_dataset[col].sum()
                plt.bar(col, len(self.full_dataset), color='lightgray', alpha=0.5, label=f'Total Samples ({len(self.full_dataset)})', width=0.5)
                bars = plt.bar(col, counts, color='tab:blue', label='Events', width=0.5)
                plt.bar_label(bars, padding=3)
                plt.title("Label Distribution")
                plt.ylabel("Counts")
                plt.xticks(rotation=45, ha='right')
                plt.xlim(-1,1)
                plt.legend()
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"full_{col.lower()}_distribution.png"))
                    plt.close()
                else:
                    plt.show()
        elif type_dataset == "train":
            self.train_dataset.plot_label_distribution("Train", save_dir=save_dir)
        elif type_dataset == "val":
            self.val_dataset.plot_label_distribution("Val", save_dir=save_dir)
        elif type_dataset == "test":
            self.test_dataset.plot_label_distribution("Test", save_dir=save_dir)
        else:
            raise ValueError("type_dataset must be 'train', 'val', 'test' or None")

    def train_dataloader(self):
        dataset = self.train_dataset
        if hasattr(dataset, 'lengths') and dataset.lengths is not None:
            sampler = SmartBatchSampler(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                ensure_positive=True
            )
            return DataLoader(
                dataset, 
                batch_sampler=sampler, 
                num_workers=self.num_workers, 
                collate_fn=dataset.collate_fn,
                pin_memory=True,      
                prefetch_factor=2,
                persistent_workers=True 
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                collate_fn=dataset.collate_fn,
                pin_memory=True,      
                prefetch_factor=2,
                persistent_workers=True 
            )

    def val_dataloader(self):
        dataset = self.val_dataset
        if hasattr(dataset, 'lengths') and dataset.lengths is not None:
            sampler = SmartBatchSampler(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                ensure_positive=False,
            )
            return DataLoader(
                dataset, 
                batch_sampler=sampler, 
                num_workers=self.num_workers, 
                collate_fn=dataset.collate_fn,
                pin_memory=True,      
                prefetch_factor=2,
                persistent_workers=True 
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                collate_fn=dataset.collate_fn,
                pin_memory=True,      
                prefetch_factor=2,
                persistent_workers=True 
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,      
            prefetch_factor=2,
            persistent_workers=True 
        )
class StratifiedMultimodalDataset(MultimodalDataset):
    """
    Extends MultimodalDataset to emit an integer age 'stratum_id' per subject
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

class StratifiedMultimodalDataModule(MultimodalDataModule):
    def __init__(self, n_bins: int = 10, **kwargs):
        self.n_bins = n_bins
        super().__init__(**kwargs)
        # Re-assign self.train_dataset, self.val_dataset, self.test_dataset
        
        columns_to_drop = []
        
        self.train_dataset = StratifiedMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.train_dataframe.drop(columns=columns_to_drop, axis=1),
        )
        self.val_dataset = StratifiedMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.val_dataframe.drop(columns=columns_to_drop, axis=1),
        )
        self.test_dataset = StratifiedMultimodalDataset(
            n_bins=self.n_bins,
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            dataframe=self.test_dataframe.drop(columns=columns_to_drop, axis=1),
        )
