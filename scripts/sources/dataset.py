import logging
import os
import sys
import pickle as pkl
import random
import re
import warnings
from pathlib import Path
from typing import Literal, Optional, Union
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.samplers import SmartBatchSampler
from utils.utils import (
    compute_lengths,
    compute_lengths_raw,
    dataframe_split_stratified,
    extract_csv,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

InputMode = Literal["features", "raw"]


def pad_batch_features(batch: list[torch.Tensor], padding_value=0.0):
    """Pad [C, T, F] feature tensors."""
    lengths = [tensor.shape[1] for tensor in batch]
    max_length = max(lengths)

    padded_list = []
    for tensor in batch:
        c, t, f = tensor.shape
        if t < max_length:
            pad = torch.full((c, max_length - t, f), padding_value, dtype=tensor.dtype)
            padded = torch.cat([tensor, pad], dim=1)
        else:
            padded = tensor
        padded_list.append(padded)

    mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]
    return torch.stack(padded_list, dim=0), mask.long(), torch.tensor(lengths)


def pad_batch_raw(batch: list[torch.Tensor], padding_value=0.0):
    """Pad [C, T] raw signal tensors."""
    lengths = [tensor.shape[1] for tensor in batch]
    max_length = max(lengths)
    batch_size = len(batch)
    c = batch[0].shape[0]

    padded_batch = torch.full((batch_size, c, max_length), padding_value, dtype=batch[0].dtype)
    for i, tensor in enumerate(batch):
        t = tensor.shape[1]
        padded_batch[i, :, :t] = tensor

    mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]
    return padded_batch, mask.long(), torch.tensor(lengths)


class MultimodalDataset(Dataset):
    def __init__(
        self,
        config_csv: dict,
        data_folder: str,
        model_config: dict,
        dataframe: pd.DataFrame = None,
        input_mode: InputMode = "features",
        sampler_per_length: bool = True,
    ):
        assert input_mode in ("features", "raw"), "input_mode must be 'features' or 'raw'"
        assert all(mod in ["eeg", "eog", "ecg"] for mod in model_config["modalities"]), (
            "Unsupported modality. Modalities must be among ['eeg', 'eog', 'ecg']."
        )

        self.input_mode = input_mode
        self.n_exam = re.search(r"shhs(\d+)", os.path.basename(config_csv["original_csv_file"])).group(1)

        if dataframe is None:
            self.df = extract_csv(**config_csv)
        else:
            self.df = dataframe

        self.data_folder = data_folder
        self.config_modalities = model_config["modalities"]

        if input_mode == "features":
            assert model_config["modalities"].keys() == model_config["resolution_input_embedding_s"].keys(), (
                "Modalities and resolution_input_embedding_s must match for features mode."
            )
            self.resolution_input_embedding_s = model_config["resolution_input_embedding_s"]
            self.resolution_output_embedding_s = model_config["resolution_output_embedding_s"]
        else:
            self.sampling_rate_hz = model_config.get("sampling_rate_hz", {})
            self.max_signal_length_s = model_config.get("max_signal_length_s", None)

        self.use_demographic_features = model_config.get("use_demographic_features", False)
        dem_var = config_csv.get("demographic_stratification_var", [])
        self.columns_demographic = [
            col for col in self.df.columns if any(keyword.lower() in col.lower() for keyword in dem_var)
        ]
        if self.use_demographic_features:
            if dem_var == []:
                warnings.warn("Demographic features enabled but no demographic variables in config.")
            self.demo_data = self.df[self.columns_demographic].copy()

        self.labels = self.df[
            [
                col
                for col in self.df.columns
                if "time" not in col and col != "id" and not any(keyword.lower() in col.lower() for keyword in dem_var)
            ]
        ]
        self.labels_names = self.labels.columns.tolist()
        self.n_labels = self.labels.shape[1]

        if self.n_labels == 1:
            self.times = self.df[["time"]]
            self.times_names = ["time"]
        else:
            self.times = self.df[[col + "_time" for col in self.labels.columns]]
            self.times_names = self.times.columns.tolist()

        self.lengths = None
        if sampler_per_length:
            if input_mode == "features":
                lengths_file = os.path.join(
                    Path(config_csv["original_csv_file"]).parent,
                    Path(config_csv["original_csv_file"]).stem + "_lengths.csv",
                )
                if not os.path.exists(lengths_file):
                    modality = list(self.resolution_input_embedding_s.keys())[0].upper()
                    resolution = list(self.resolution_input_embedding_s.values())[0]
                    compute_lengths(config_csv["original_csv_file"], data_folder, modality, resolution)
            else:
                lengths_file = os.path.join(
                    Path(config_csv["original_csv_file"]).parent,
                    Path(config_csv["original_csv_file"]).stem + "_lengths_raw.csv",
                )
                if not os.path.exists(lengths_file):
                    compute_lengths_raw(config_csv["original_csv_file"], data_folder)

            lengths_df = pd.read_csv(lengths_file)
            lengths_map = dict(zip(lengths_df["id"], lengths_df["length"]))
            self.lengths = [lengths_map.get(uid, 0) for uid in self.df["id"]] if not self.df.empty else None

    def label_distribution(self):
        return {col: self.labels[col].value_counts().to_dict() for col in self.labels.columns}

    def plot_label_distribution(self, subtitle: Optional[str] = None, save_dir: Optional[str] = None):
        if len(self.labels.columns) > 1:
            plt.figure(figsize=(12, 6))
            counts = (self.labels == 1).sum()
            plt.bar(counts.index, [len(self)] * len(counts), color="lightgray", alpha=0.5, label=f"Total ({len(self)})")
            bars = plt.bar(counts.index, counts.values, color="tab:blue", label="Events")
            plt.bar_label(bars, padding=3)
            plt.title(f"Label Distribution - {subtitle}")
            plt.ylabel("Counts")
            plt.xticks(rotation=45, ha="right")
            plt.legend()
        else:
            col = self.labels.columns[0]
            plt.figure(figsize=(10, 6))
            counts = (self.labels[col] == 1).sum()
            plt.bar(col, len(self), color="lightgray", alpha=0.5, label=f"Total ({len(self)})", width=0.5)
            bars = plt.bar(col, counts, color="tab:blue", label="Events", width=0.5)
            plt.bar_label(bars, padding=3)
            plt.title(f"Label Distribution - {subtitle}")
            plt.ylabel("Counts")
            plt.xticks(rotation=45, ha="right")
            plt.xlim(-1, 1)
            plt.legend()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            name = f"{subtitle.lower() + '_' if subtitle else ''}distribution.png"
            plt.savefig(os.path.join(save_dir, name))
            plt.close()
        else:
            plt.show()

    def __len__(self):
        return len(self.df)

    def _load_features(self, row) -> dict:
        multimodal_data = {}
        for modality, channels in self.config_modalities.items():
            modality_data = []
            for channel in channels:
                res = (
                    f"res_{self.resolution_input_embedding_s[modality]}s"
                    if self.resolution_input_embedding_s[modality] is not None
                    else ""
                )
                data_path = os.path.join(
                    self.data_folder,
                    "features",
                    modality.upper(),
                    res,
                    channel.upper(),
                    f"shhs{self.n_exam}-{int(row['id'])}.npy",
                )
                if not os.path.exists(data_path):
                    raise ValueError(f"Data path {data_path} not found.")

                embedding = np.load(data_path)
                factor = int(self.resolution_output_embedding_s / self.resolution_input_embedding_s[modality])
                embedding = embedding[: len(embedding) - len(embedding) % factor, :]
                modality_data.append(torch.tensor(embedding, dtype=torch.float32).squeeze())

            if modality_data:
                multimodal_data[modality] = torch.stack(modality_data, dim=0)
        return multimodal_data

    def _load_raw(self, row) -> dict:
        multimodal_data = {}
        data_path = os.path.join(self.data_folder, "raw", f"shhs{self.n_exam}-{int(row['id'])}.hdf5")
        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} not found.")

        with h5py.File(data_path, "r") as hf:
            available_keys = list(hf.keys())
            used_keys = set()
            channel_mapping = {}

            for modality, channels in self.config_modalities.items():
                modality_data = []
                channel_mapping[modality] = []
                for channel in channels:
                    matched_key = None
                    for key in available_keys:
                        if channel.lower() in key.lower() and key not in used_keys:
                            matched_key = key
                            used_keys.add(key)
                            channel_mapping[modality].append(f"{channel} -> {matched_key}")
                            break
                    if matched_key is None:
                        raise ValueError(
                            f"Could not find channel '{channel}' in {data_path}. Available: {available_keys}"
                        )

                    tensor = torch.tensor(hf[matched_key][:], dtype=torch.float32).squeeze()
                    if self.max_signal_length_s is not None:
                        fs = self.sampling_rate_hz.get(modality, 128)
                        max_samples = int(fs * self.max_signal_length_s)
                        if tensor.shape[-1] > max_samples:
                            tensor = tensor[..., :max_samples]
                    modality_data.append(tensor)

                if modality_data:
                    multimodal_data[modality] = torch.stack(modality_data, dim=0)

        return multimodal_data

    def _handle_corrupt(self, patient_id, idx: int, data_path: str, error: Exception):
        logger.error(f"Error loading data from {data_path}: {error}")
        corrupt_cache_path = os.path.join(self.data_folder, "corrupted_files.txt")
        try:
            with open(corrupt_cache_path, "a") as f:
                f.write(f"{patient_id}\n")
        except Exception as cache_error:
            logger.error(f"Failed to write corrupt cache: {cache_error}")

        if len(self.df) > 1:
            new_idx = random.randint(0, len(self.df) - 1)
            while new_idx == idx:
                new_idx = random.randint(0, len(self.df) - 1)
            logger.warning(f"Skipping corrupted id {patient_id}, resampling index {new_idx}.")
            return self.__getitem__(new_idx)
        raise RuntimeError(f"Corrupted file: {data_path}, no alternative samples available.") from error

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        data_path = (
            os.path.join(self.data_folder, "raw", f"shhs{self.n_exam}-{int(row['id'])}.hdf5")
            if self.input_mode == "raw"
            else f"shhs{self.n_exam}-{int(row['id'])}
        )

        try:
            multimodal_data = self._load_raw(row) if self.input_mode == "raw" else self._load_features(row)
        except Exception as error:
            return self._handle_corrupt(row["id"], idx, data_path, error)

        labels = torch.tensor(self.labels.iloc[idx, :].values, dtype=torch.float32)
        times = torch.tensor(self.times.iloc[idx, :].values, dtype=torch.float32)
        labels = torch.stack((labels, times), dim=1)

        demo_tensor = torch.empty(0)
        if self.use_demographic_features:
            demo_values = self.demo_data.iloc[idx].values.astype(float)
            demo_tensor = torch.tensor(demo_values, dtype=torch.float32)

        return multimodal_data, labels, demo_tensor, torch.tensor(int(row["id"]))

    def collate_fn(self, batch, padding_value=0.0):
        modalities = list(batch[0][0].keys())
        inputs, masks = {}, {}

        labels = torch.stack([sample[1] for sample in batch], dim=0)
        ids = torch.stack([sample[-1] for sample in batch], dim=0)

        demo_features = None
        if batch[0][2].numel() > 0:
            demo_features = torch.stack([sample[2] for sample in batch], dim=0)

        pad_fn = pad_batch_raw if self.input_mode == "raw" else pad_batch_features
        for modality in modalities:
            tensor_list = [sample[0][modality] for sample in batch]
            padded_tensors, mask, _ = pad_fn(tensor_list, padding_value=padding_value)
            inputs[modality] = padded_tensors
            masks[modality] = mask

        return inputs, masks, labels, demo_features, ids


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config_csv: dict,
        data_folder: str,
        model_config: dict,
        input_mode: InputMode = "features",
        batch_size: int = 32,
        num_workers: int = 4,
        scaler: Optional[StandardScaler] = None,
        use_cache: bool = True,
        random_state: int = 42,
        test: Union[float, list] = 0.2,
        val: Union[float, list] = 0.2,
        save_splits: bool = False,
    ):
        super().__init__()
        self.config_csv = config_csv
        self.data_folder = data_folder
        self.model_config = model_config
        self.input_mode = input_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test = test
        self.val = val
        self.random_state = random_state
        self.save_splits = save_splits
        self.use_cache = use_cache
        self.scaler = scaler

        self.full_dataset = extract_csv(**self.config_csv)

        corrupt_cache_path = os.path.join(self.data_folder, "corrupted_files.txt")
        if os.path.exists(corrupt_cache_path):
            with open(corrupt_cache_path, "r") as f:
                corrupt_ids = {line.strip() for line in f if line.strip()}
            if corrupt_ids:
                initial_len = len(self.full_dataset)
                self.full_dataset = self.full_dataset[~self.full_dataset["id"].astype(str).isin(corrupt_ids)]
                if len(self.full_dataset) < initial_len:
                    logger.warning(f"Removed {initial_len - len(self.full_dataset)} corrupted files from dataset.")

        self.dem_var = self.config_csv.get("demographic_stratification_var", [])
        self.demographic_columns = [
            col for col in self.full_dataset.columns if any(kw.lower() in col.lower() for kw in self.dem_var)
        ]
        self.labels_names = [
            col
            for col in self.full_dataset.columns
            if "time" not in col and col not in ("id", "time") and col not in self.demographic_columns
        ]
        self.n_labels = len(self.labels_names)
        self.task = "survival_binary" if self.n_labels == 1 else "survival_multilabel"

        self.dataset_events_name = Path(self.config_csv["original_csv_file"]).stem.split("from_")[1].split("_")[0]
        self.reproducibility_path = (
            os.path.join(
                Path(self.config_csv["original_csv_file"]).parent,
                "reproducibility",
                self.dataset_events_name,
                "_".join(self.dem_var),
            )
            if "reproducibility" not in str(Path(self.config_csv["original_csv_file"]).parent)
            else Path(self.config_csv["original_csv_file"]).parent
        )
        self.json_splits_file_name = Path(self.config_csv["original_csv_file"]).stem + "_split.json"

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

        if self.demographic_columns:
            self._fit_demographics_scaler(scaler)

        dataset_kwargs = dict(
            config_csv=self.config_csv,
            data_folder=self.data_folder,
            model_config=self.model_config,
            input_mode=self.input_mode,
        )
        self.train_dataset = MultimodalDataset(dataframe=self.train_dataframe, **dataset_kwargs)
        self.val_dataset = MultimodalDataset(dataframe=self.val_dataframe, **dataset_kwargs)
        self.test_dataset = MultimodalDataset(dataframe=self.test_dataframe, **dataset_kwargs)

    def _fit_demographics_scaler(self, scaler):
        self.scaled_columns = []
        for col in self.demographic_columns:
            for df in (self.train_dataframe, self.val_dataframe, self.test_dataframe):
                if not df.empty and col in df and df[col].value_counts().shape[0] == 2:
                    df.loc[:, col] = df.loc[:, col] - 1 if df[col].max() == 2 else df.loc[:, col]
                elif not df.empty and col in df and df[col].value_counts().shape[0] > 2:
                    df[col] = df[col].astype(float)
            if (
                not self.train_dataframe.empty
                and col in self.train_dataframe
                and self.train_dataframe[col].value_counts().shape[0] > 2
            ):
                self.scaled_columns.append(col)

        scaler_path = os.path.join(self.reproducibility_path, "demographics_scaler.pkl")
        if scaler is not None:
            self.scaler = scaler
        elif os.path.exists(scaler_path):
            logger.info(f"Loading demographics scaler from: {scaler_path}")
            self.scaler = pkl.load(open(scaler_path, "rb"))
            logger.info(f'Means: {self.scaler.mean_}')
            logger.info(f'Vars: {self.scaler.var_}')
            logger.info(f'Samples: {self.scaler.n_samples_seen_}')
        elif self.scaled_columns:
            logger.info(f"Fitting new demographics scaler with {self.scaled_columns}")
            self.scaler = StandardScaler()
            self.scaler.fit(self.train_dataframe.loc[:, self.scaled_columns])
            os.makedirs(self.reproducibility_path, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pkl.dump(self.scaler, f)
            logger.info(f'Means: {self.scaler.mean_}')
            logger.info(f'Vars: {self.scaler.var_}')
            logger.info(f'Samples: {self.scaler.n_samples_seen_}')

        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            self.scaled_columns = list(self.scaler.feature_names_in_)

        if self.scaler is not None and self.scaled_columns:
            for df in (self.train_dataframe, self.val_dataframe, self.test_dataframe):
                if not df.empty:
                    df.loc[:, self.scaled_columns] = self.scaler.transform(df.loc[:, self.scaled_columns])

    def label_distribution(self, type_dataset: Optional[str] = None):
        if type_dataset is None:
            return {col: self.full_dataset[col].value_counts().to_dict() for col in self.labels_names}
        return getattr(self, f"{type_dataset}_dataset").label_distribution()

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

    def _make_dataloader(self, dataset, shuffle: bool, ensure_positive: bool):
        if hasattr(dataset, "lengths") and dataset.lengths is not None:
            sampler = SmartBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                ensure_positive=ensure_positive,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True, ensure_positive=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, shuffle=True, ensure_positive=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataloader, shuffle=False, ensure_positive=False) 