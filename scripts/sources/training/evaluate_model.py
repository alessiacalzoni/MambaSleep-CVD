"""
Comprehensive Model Evaluation Script for PSG Cardiovascular Project

This script evaluates trained models (classification or survival) on specified datasets
and calculates per-label metrics for multilabel tasks.

Usage:
    python -m scripts.sources.evaluate_model \
        --checkpoint path/to/checkpoint.ckpt \
        --config path/to/config.yaml \
        --dataset test \
        --output_dir evaluation_results \
        --save_predictions
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sources.dataset import MultimodalDataModule
from scripts.sources.models.models import LitMultimodalModel, LitDemoBaselineModel
from scripts.sources.utils.utils import update_dict
from scripts.sources.base_system.loss import ClassificationLoss, SurvivalLoss

# Import survival metrics
# Import survival metrics
try:
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.brier_score import BrierScore
    TORCHSURV_AVAILABLE = True
except ImportError:
    TORCHSURV_AVAILABLE = False
    logging.warning("torchsurv not available. Survival metrics will use fallback implementations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def from_path_to_ids(paths):
    return [int(Path(path).stem.split('-')[1]) for path in paths]


class ModelEvaluator:
    """Evaluator class for both classification and survival models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict,
        only_demo: bool=False,
        dataset_split: str = "test",
        output_dir: str = "evaluation_results",
        save_predictions: bool = False,
        auroc_time_points: Optional[List[float]] = None,
        n_bootstraps: int = 0,
        ci_level: float = 0.95,
        age_metrics: list = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            config: Configuration dictionary (from YAML)
            dataset_split: Which split to evaluate ('test', 'val', 'train')
            output_dir: Directory to save evaluation results
            save_predictions: Whether to save model predictions
            auroc_time_points: List of time points (in years) for time-dependent AUROC calculation in survival tasks
            n_bootstraps: Number of bootstrap samples (0 for no bootstrapping)
            ci_level: Confidence interval level (default: 0.95)
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.dataset_split = dataset_split
        self.output_dir = Path(output_dir)
        self.save_predictions = save_predictions
        self.auroc_time_points = auroc_time_points
        self.n_bootstraps = n_bootstraps
        self.ci_level = ci_level
        self.only_demo = only_demo
        self.age_metrics = age_metrics
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Will be set during initialization
        self.model = None
        self.datamodule = None
        self.task_type = None
        self.label_names = None
        
    def load_checkpoint_and_config(self):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        n_labels = self.datamodule.n_labels
        loss = SurvivalLoss() if self.task_type == 'survival' else ClassificationLoss()
        
        # Load model
        if self.only_demo:
            self.model = LitDemoBaselineModel(
                n_demo_features = len(self.config['csv_config']['demographic_stratification_var'])+1 if 'med' in self.config['csv_config']['demographic_stratification_var'] else len(self.config['csv_config']['demographic_stratification_var']),
                model_config = self.config['model_config'], 
                number_labels = self.datamodule.n_labels, 
                lr=self.config['training_config'].get('lr'), 
                weight_decay=self.config['training_config'].get('weight_decay'),
                loss=loss
            )
        else:
            self.model = LitMultimodalModel(
                model_config = self.config['model_config'], 
                number_labels = self.datamodule.n_labels, 
                lr=self.config['training_config'].get('lr'), 
                weight_decay=self.config['training_config'].get('weight_decay'),
                loss=loss
            )
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Model: {self.model}")
        logger.info(f"Model loaded successfully from checkpoint")
        
    def setup_dataset(self):
        """Initialize dataset based on configuration."""
        logger.info("Setting up dataset...")

        if isinstance(self.config['training_config'].get('split_path', None), str):
            with open(self.config['training_config'].get('split_path'), 'r') as f:
                self.output_dir = Path(os.path.join(self.output_dir, Path(self.config['training_config'].get('split_path')).stem))
                os.makedirs(self.output_dir, exist_ok=True)
                splits = json.load(f)
                if 'test' in splits.keys():
                    test_ids = splits['test']
                    if isinstance(test_ids[0], str):
                        test_ids = from_path_to_ids(test_ids)
                else:
                    test_ids = None
                if 'val' in splits.keys():
                    val_ids = splits['val']
                    if isinstance(val_ids[0], str):
                        val_ids = from_path_to_ids(val_ids)
                else:
                    val_ids = None
                if 'secondary_prev' in splits.keys():
                    test_ids = splits['secondary_prev']
                    logger.info("Using secondary_prev as test")
                    if isinstance(test_ids[0], str):
                        test_ids = from_path_to_ids(test_ids)
                    val_ids = splits['primary_prev']
                    logger.info("Using primary_prev as val")
                    if isinstance(val_ids[0], str):
                        val_ids = from_path_to_ids(val_ids)
                if 'not_train' in splits.keys():
                    test_ids = splits['not_train']
                    logger.info("Using not_train as test")
                    if isinstance(test_ids[0], str):
                        test_ids = from_path_to_ids(test_ids)
                    val_ids = splits['train']
                    logger.info("Using train as val")
                    if isinstance(val_ids[0], str):
                        val_ids = from_path_to_ids(val_ids)
        
        # Initialize datamodule
        self.datamodule = MultimodalDataModule(
            config_csv=self.config['csv_config'],
            data_folder=self.config['paths']['data_folder'],
            model_config=self.config['model_config'],
            batch_size=1,  # Use batch_size=1 for evaluation to avoid any issues
            num_workers=0,
            test=test_ids if test_ids is not None else self.config['training_config'].get('test', 0.2),
            val=val_ids if val_ids is not None else self.config['training_config'].get('val', 0.2),
            random_state=self.config['training_config'].get('random_state', 42),
        )
        
        # Get task type and label names
        self.task_type = self.datamodule.task
        self.label_names = self.datamodule.labels_names
        
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Number of labels: {len(self.label_names)}")
        logger.info(f"Label names: {self.label_names}")
        
    def get_dataloader(self) -> DataLoader:
        """Get the appropriate dataloader based on dataset_split."""
        if self.dataset_split == "test":
            return DataLoader(self.datamodule.test_dataset, batch_size=1, shuffle=False, collate_fn = self.datamodule.test_dataset.collate_fn)
        elif self.dataset_split == "val":
            return DataLoader(self.datamodule.val_dataset, batch_size=1, shuffle=False, collate_fn = self.datamodule.val_dataset.collate_fn)
        elif self.dataset_split == "train":
            return DataLoader(self.datamodule.train_dataset, batch_size=1, shuffle=False, collate_fn = self.datamodule.train_dataset.collate_fn)
        else:
            raise ValueError(f"Invalid dataset_split: {self.dataset_split}")
    
    def collect_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect model predictions and ground truth labels.
        
        Returns:
            predictions: Model outputs (logits/hazards) - shape (N, num_labels)
            labels: Ground truth labels - shape (N, num_labels) or (N, num_labels, 2) for survival
            probabilities: Probabilities (for classification only) - shape (N, num_labels)
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        all_ages = []
        all_ids = []

        demo_df = pd.read_csv(self.config['csv_config']['demographic_csv_file'])
        age_col = [col for col in demo_df.columns if 'age_s' in col][0]
        
        logger.info(f"Collecting predictions from {self.dataset_split} set...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if len(batch) == 5:
                    inputs, masks, labels, demo_features, ids = batch
                else:
                    inputs, masks, labels, ids = batch
                    demo_features = None
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                masks = {k: (v.to(self.device) if v is not None else None) for k, v in masks.items()}
                labels = labels.to(self.device)
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, masks, demo_features)
                
                # Store predictions and labels
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_ages.append(demo_df[demo_df['nsrrid'].isin(ids.cpu().tolist())][age_col].values)
                all_ids.extend(ids.cpu().tolist())
                
                # Calculate probabilities for classification
                if "classification" in self.task_type:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_probs.append(probs)
        
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        ages = np.concatenate(all_ages, axis = 0)
        ids_array = np.array(all_ids)
        
        if "classification" in self.task_type:
            probabilities = np.concatenate(all_probs, axis=0)
        else:
            probabilities = None
            
        logger.info(f"Collected {len(predictions)} samples")
        return predictions, labels, probabilities, ages, ids_array
    
    def evaluate_classification(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Calculate classification metrics per label.
        
        Args:
            predictions: Logits - shape (N, num_labels)
            labels: Ground truth - shape (N, num_labels)
            probabilities: Probabilities - shape (N, num_labels)
            
        Returns:
            Dictionary with per-label and averaged metrics
        """
        logger.info("Calculating classification metrics...")
        
        num_labels = labels.shape[1]
        results = {
            "task_type": self.task_type,
            "num_samples": len(labels),
            "num_labels": num_labels,
            "per_label_metrics": {},
            "macro_averaged_metrics": {},
            "micro_averaged_metrics": {},
        }
        
        # Per-label metrics
        for i, label_name in enumerate(self.label_names):
            label_true = labels[:, i]
            label_probs = probabilities[:, i]
            label_preds = (label_probs > 0.5).astype(int)
            
            # Check if we have both classes
            if len(np.unique(label_true)) < 2:
                logger.warning(f"Label {label_name} has only one class in the dataset. Skipping some metrics.")
                auroc = None
                auprc = None
            else:
                try:
                    auroc = roc_auc_score(label_true, label_probs)
                    auprc = average_precision_score(label_true, label_probs)
                except Exception as e:
                    logger.warning(f"Error calculating AUROC/AUPRC for {label_name}: {e}")
                    auroc = None
                    auprc = None
            
            # Calculate other metrics
            accuracy = accuracy_score(label_true, label_preds)
            precision = precision_score(label_true, label_preds, zero_division=0)
            recall = recall_score(label_true, label_preds, zero_division=0)
            f1 = f1_score(label_true, label_preds, zero_division=0)
            
            # Count positive samples
            num_positive = int(label_true.sum())
            num_negative = int((1 - label_true).sum())
            
            results["per_label_metrics"][label_name] = {
                "auroc": float(auroc) if auroc is not None else None,
                "auprc": float(auprc) if auprc is not None else None,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "num_positive": num_positive,
                "num_negative": num_negative,
            }
        
        # Macro-averaged metrics (average of per-label metrics)
        valid_aurocs = [m["auroc"] for m in results["per_label_metrics"].values() if m["auroc"] is not None]
        valid_auprcs = [m["auprc"] for m in results["per_label_metrics"].values() if m["auprc"] is not None]
        
        results["macro_averaged_metrics"] = {
            "auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "auprc": float(np.mean(valid_auprcs)) if valid_auprcs else None,
            "accuracy": float(np.mean([m["accuracy"] for m in results["per_label_metrics"].values()])),
            "precision": float(np.mean([m["precision"] for m in results["per_label_metrics"].values()])),
            "recall": float(np.mean([m["recall"] for m in results["per_label_metrics"].values()])),
            "f1_score": float(np.mean([m["f1_score"] for m in results["per_label_metrics"].values()])),
        }
        
        # Micro-averaged metrics (treat all labels as one)
        if num_labels > 1:
            labels_flat = labels.flatten()
            probs_flat = probabilities.flatten()
            preds_flat = (probs_flat > 0.5).astype(int)
            
            try:
                micro_auroc = roc_auc_score(labels_flat, probs_flat)
                micro_auprc = average_precision_score(labels_flat, probs_flat)
            except:
                micro_auroc = None
                micro_auprc = None
                
            results["micro_averaged_metrics"] = {
                "auroc": float(micro_auroc) if micro_auroc is not None else None,
                "auprc": float(micro_auprc) if micro_auprc is not None else None,
                "accuracy": float(accuracy_score(labels_flat, preds_flat)),
                "precision": float(precision_score(labels_flat, preds_flat, zero_division=0)),
                "recall": float(recall_score(labels_flat, preds_flat, zero_division=0)),
                "f1_score": float(f1_score(labels_flat, preds_flat, zero_division=0)),
            }
        
        return results
    

    
    def evaluate_survival(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """
        Calculate survival metrics per label.
        
        Args:
            predictions: Hazards - shape (N, num_labels)
            labels: Ground truth with (is_event, time) - shape (N, num_labels, 2)
            
        Returns:
            Dictionary with per-label and averaged metrics
        """
        logger.info("Calculating survival metrics...")
        
        num_labels = labels.shape[1]
        results = {
            "task_type": self.task_type,
            "num_samples": len(labels),
            "num_labels": num_labels,
            "per_label_metrics": {},
            "macro_averaged_metrics": {},
            "micro_averaged_metrics": {},
        }
        
        # Add time points to results if specified
        if self.auroc_time_points:
            results["auroc_time_points_years"] = self.auroc_time_points
        
        # Extract event indicators and times
        is_event = labels[:, :, 0]  # (N, num_labels)
        event_times = labels[:, :, 1]  # (N, num_labels)
        logger.info(f"Min times: {np.min(event_times)}")
        for i in range(num_labels):
            if is_event[:, i].astype(bool).sum() == 0:
                logger.info(f"No events for label {self.label_names[i]}")
                continue
            logger.info(f"Min events times for label {self.label_names[i]}: {np.min(event_times[:, i][is_event[:, i].astype(bool)])/365:.2f} years")
        
        # Per-label metrics
        for i, label_name in enumerate(self.label_names):
            label_hazards = predictions[:, i]
            label_is_event = is_event[:, i]
            label_event_times = event_times[:, i]
            
            # Mask out unobserved events
            valid_mask = label_is_event >= 0
            if not np.any(valid_mask):
                num_events = 0
                num_censored = 0
                label_hazards = np.array([])
                label_is_event = np.array([])
                label_event_times = np.array([])
            else:
                label_hazards = label_hazards[valid_mask]
                label_is_event = label_is_event[valid_mask]
                label_event_times = label_event_times[valid_mask]
                
                # Count events
                num_events = int(label_is_event.sum())
                num_censored = int((1 - label_is_event).sum())
            
            # Skip C-Index and AUROC if no events (all censored) for this label
            if num_events == 0:
                logger.warning(f"Label '{label_name}' has 0 events (all censored). "
                               f"C-Index and AUROC cannot be computed — setting to None.")
                label_metrics = {
                    "c_index": None,
                    "num_events": 0,
                    "num_censored": num_censored,
                }
                if self.auroc_time_points:
                    for time_point in self.auroc_time_points:
                        label_metrics[f"auroc_{time_point}y"] = None
                results["per_label_metrics"][label_name] = label_metrics
                continue
            
            # Calculate C-Index
            if TORCHSURV_AVAILABLE:
                try:
                    cindex_metric = ConcordanceIndex()
                    hazards_tensor = torch.tensor(label_hazards, dtype=torch.float32)
                    is_event_tensor = torch.tensor(label_is_event, dtype=torch.bool)
                    event_times_tensor = torch.tensor(label_event_times, dtype=torch.float32)
                    
                    cindex = cindex_metric(hazards_tensor, is_event_tensor, event_times_tensor)
                    cindex = float(cindex.item())
                except Exception as e:
                    logger.warning(f"TorchSurv C-Index calculation failed for {label_name}: {e}. Returning None.")
                    cindex = None
            else:
                logger.warning("TorchSurv not available. Skipping C-Index calculation.")
                cindex = None
            
            # Initialize metrics dictionary
            c_index_val = float(cindex) if cindex is not None else None
            label_metrics = {
                "c_index": c_index_val,
                "num_events": num_events,
                "num_censored": num_censored,
            }
            
            # Calculate time-dependent AUROC if time points specified
            if self.auroc_time_points and TORCHSURV_AVAILABLE:
                auroc_at_times = {}
                # Calculate min and max times in days from the current sample
                min_time_days = np.min(event_times)
                max_time_days = np.max(event_times)
                
                for time_point in self.auroc_time_points:
                    time_point_days = int(time_point * 365)
                    
                    if time_point_days < min_time_days or time_point_days > max_time_days:
                         logger.warning(f"Time point {time_point} years ({time_point_days} days) is outside data range [{min_time_days:.1f}, {max_time_days:.1f}]. Skipping.")
                         auroc_at_times[f"auroc_{time_point}y"] = None
                         continue

                    try:
                        auc_metric = Auc()
                        auroc_value = auc_metric(
                            estimate=hazards_tensor,
                            time=event_times_tensor,
                            event=is_event_tensor,
                            new_time=int(time_point)*365
                        )
                        auroc_at_times[f"auroc_{time_point}y"] = float(auroc_value.item())
                    except Exception as e:
                        logger.warning(f"Failed to calculate AUROC at {time_point} years for {label_name}: {e}")
                        auroc_at_times[f"auroc_{time_point}y"] = None
                
                label_metrics.update(auroc_at_times)
            elif self.auroc_time_points and not TORCHSURV_AVAILABLE:
                logger.warning("Time-dependent AUROC requested but torchsurv is not available. Skipping AUROC calculation.")
            
            results["per_label_metrics"][label_name] = label_metrics
        
        # Macro-averaged metrics (only average over labels with valid c_index)
        valid_cindexes = [m["c_index"] for m in results["per_label_metrics"].values() if m["c_index"] is not None]
        macro_metrics = {
            "c_index": float(np.mean(valid_cindexes)) if valid_cindexes else None,
        }
        
        # Add macro-averaged AUROC at each time point
        if self.auroc_time_points:
            for time_point in self.auroc_time_points:
                auroc_key = f"auroc_{time_point}y"
                valid_aurocs = [m[auroc_key] for m in results["per_label_metrics"].values() if auroc_key in m and m[auroc_key] is not None]
                if valid_aurocs:
                    macro_metrics[auroc_key] = float(np.mean(valid_aurocs))
                else:
                    macro_metrics[auroc_key] = None
        
        results["macro_averaged_metrics"] = macro_metrics
        
        # Micro-averaged metrics (flatten all labels)
        if num_labels > 1:
            valid_mask = is_event >= 0
            hazards_flat = predictions[valid_mask]
            is_event_flat = is_event[valid_mask]
            event_times_flat = event_times[valid_mask]
            
            if len(is_event_flat) == 0:
                 micro_cindex = None
                 micro_metrics = {"c_index": None}
                 if self.auroc_time_points:
                     for time_point in self.auroc_time_points:
                         micro_metrics[f"auroc_{time_point}y"] = None
                 results["micro_averaged_metrics"] = micro_metrics
                 return results

            if TORCHSURV_AVAILABLE:
                try:
                    cindex_metric = ConcordanceIndex()
                    hazards_tensor_flat = torch.tensor(hazards_flat, dtype=torch.float32)
                    is_event_tensor_flat = torch.tensor(is_event_flat, dtype=torch.bool)
                    event_times_tensor_flat = torch.tensor(event_times_flat, dtype=torch.float32)
                    micro_cindex = cindex_metric(hazards_tensor_flat, is_event_tensor_flat, event_times_tensor_flat)
                    micro_cindex = float(micro_cindex.item())
                except Exception as e:
                     logger.warning(f"Micro C-Index calculation failed: {e}")
                     micro_cindex = None
            else:
                 micro_cindex = None
            
            micro_metrics = {}
            if micro_cindex is not None:
                micro_metrics["c_index"] = float(micro_cindex)
            else:
                 micro_metrics["c_index"] = None
            
            # Add micro-averaged AUROC at each time point
            if self.auroc_time_points and TORCHSURV_AVAILABLE:
                hazards_tensor_flat = torch.tensor(hazards_flat, dtype=torch.float32)
                is_event_tensor_flat = torch.tensor(is_event_flat, dtype=torch.bool)
                event_times_tensor_flat = torch.tensor(event_times_flat, dtype=torch.float32)
                
                for time_point in self.auroc_time_points:
                    try:
                        auc_metric = Auc()
                        auroc_value = auc_metric(
                            estimate=hazards_tensor_flat,
                            time=event_times_tensor_flat,
                            event=is_event_tensor_flat,
                            new_time=int(time_point)*365
                        )
                        micro_metrics[f"auroc_{time_point}y"] = float(auroc_value.item())
                    except Exception as e:
                        logger.warning(f"Failed to calculate micro AUROC at {time_point} years: {e}")
                        micro_metrics[f"auroc_{time_point}y"] = None
            
            results["micro_averaged_metrics"] = micro_metrics
        
        return results
    
    def save_results(self, results: Dict):
        """Save evaluation results to JSON and CSV."""
        save_name = "primary_prev" if self.dataset_split == "val" else ("secondary_prev" if self.dataset_split == "test" else self.dataset_split)
        
        # Save JSON
        json_path = self.output_dir / f"evaluation_results_{save_name}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {json_path}")
        
        # Save CSV (per-label metrics)
        csv_path = self.output_dir / f"evaluation_results_{save_name}.csv"
        
        # Create DataFrame from per-label metrics
        df_data = []
        for label_name, metrics in results["per_label_metrics"].items():
            row = {"label": label_name}
            row.update(metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Per-label metrics saved to: {csv_path}")
        
        # Save averaged metrics
        avg_csv_path = self.output_dir / f"evaluation_averaged_{save_name}.csv"
        avg_data = []
        
        for avg_type in ["macro_averaged_metrics", "micro_averaged_metrics"]:
            if avg_type in results and results[avg_type]:
                row = {"average_type": avg_type.replace("_metrics", "")}
                row.update(results[avg_type])
                avg_data.append(row)
        
        if avg_data:
            avg_df = pd.DataFrame(avg_data)
            avg_df.to_csv(avg_csv_path, index=False)
            logger.info(f"Averaged metrics saved to: {avg_csv_path}")
    
    def save_predictions_to_file(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        ages: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None
    ):
        """Save model predictions to file."""
        save_name = "primary_prev" if self.dataset_split == "val" else ("secondary_prev" if self.dataset_split == "test" else self.dataset_split)
        pred_path = self.output_dir / f"predictions_{save_name}.npz"
        
        save_dict = {
            "predictions": predictions,
            "labels": labels,
        }
        
        if probabilities is not None:
            save_dict["probabilities"] = probabilities
        if ages is not None:
            save_dict["ages"] = ages
        if ids is not None:
            save_dict["ids"] = ids
        
        np.savez(pred_path, **save_dict)
        logger.info(f"Predictions saved to: {pred_path}")
    
    def compute_confidence_intervals(self, bootstrap_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Compute confidence intervals from bootstrap metrics.
        
        Returns nested dict: {metric_name: {mean, std, ci_lower, ci_upper}}
        """
        # Aggregate metrics
        agg_metrics = {}
        
        # Initialize structure based on first sample
        if not bootstrap_metrics:
            return {}
            
        # Helper to recursively traverse and collect values
        def collect_values(metrics_list, key_path=[]):
            collected = {}
            
            # Get structure from first item
            structure = metrics_list[0]
            for safe_key in key_path:
                structure = structure[safe_key]
                
            for k, v in structure.items():
                if isinstance(v, (int, float)) and v is not None:
                    # Leaf node with numeric value
                    values = []
                    for m in metrics_list:
                        val = m
                        for p in key_path:
                            val = val[p]
                        if val[k] is not None:
                            values.append(val[k])
                    
                    if values:
                        collected[k] = values
                elif isinstance(v, dict):
                    # Recurse
                    nested = collect_values(metrics_list, key_path + [k])
                    collected[k] = nested
            
            return collected

        # Helper to compute stats on collected values
        def compute_stats(collected_data):
            stats_result = {}
            for k, v in collected_data.items():
                if isinstance(v, list):
                    # It's a list of values, compute stats
                    values = np.array(v)
                    mean = np.mean(values)
                    std = np.std(values)
                    
                    alpha = 1.0 - self.ci_level
                    lower = np.percentile(values, alpha / 2.0 * 100)
                    upper = np.percentile(values, (1.0 - alpha / 2.0) * 100)
                    
                    stats_result[f"{k}_mean"] = float(mean)
                    stats_result[f"{k}_std"] = float(std)
                    stats_result[f"{k}_ci_lower"] = float(lower)
                    stats_result[f"{k}_ci_upper"] = float(upper)
                elif isinstance(v, dict):
                    # Recurse
                    stats_result[k] = compute_stats(v)
            return stats_result

        # 1. Collect all values
        # We need to handle per_label_metrics, macro_averaged_metrics, micro_averaged_metrics
        # Flattening simple strategy: Iterate over top-level keys of interest
        
        results_stats = {}
        
        for section in ["per_label_metrics", "macro_averaged_metrics", "micro_averaged_metrics"]:
            if section in bootstrap_metrics[0]:
                if section == "per_label_metrics":
                    stats_result = {}
                    for label in self.label_names:
                        # Collect list of dicts for this label
                        label_values = []
                        for bm in bootstrap_metrics:
                             if label in bm[section]:
                                 label_values.append(bm[section][label])
                        
                        if label_values:
                             # Provide a helper that just takes list of dicts and returns dict of list of vals
                             # Actually we can just iterate keys of the first one
                             label_stats = {}
                             keys = label_values[0].keys()
                             for metric_key in keys:
                                 vals = [lv[metric_key] for lv in label_values if lv.get(metric_key) is not None]
                                 if vals:
                                     vals = np.array(vals)
                                     alpha = 1.0 - self.ci_level
                                     
                                     label_stats[f"{metric_key}_mean"] = float(np.mean(vals))
                                     label_stats[f"{metric_key}_std"] = float(np.std(vals))
                                     label_stats[f"{metric_key}_ci_lower"] = float(np.percentile(vals, alpha / 2.0 * 100))
                                     label_stats[f"{metric_key}_ci_upper"] = float(np.percentile(vals, (1.0 - alpha / 2.0) * 100))
                             
                             stats_result[label] = label_stats
                    results_stats[section] = stats_result

                else:
                    # Averaged metrics (flat dict)
                    params_list = [bm[section] for bm in bootstrap_metrics if section in bm]
                    if params_list:
                        stats_result = {}
                        keys = params_list[0].keys()
                        for metric_key in keys:
                             vals = [p[metric_key] for p in params_list if p.get(metric_key) is not None]
                             if vals:
                                 vals = np.array(vals)
                                 alpha = 1.0 - self.ci_level
                                 
                                 stats_result[f"{metric_key}_mean"] = float(np.mean(vals))
                                 stats_result[f"{metric_key}_std"] = float(np.std(vals))
                                 stats_result[f"{metric_key}_ci_lower"] = float(np.percentile(vals, alpha / 2.0 * 100))
                                 stats_result[f"{metric_key}_ci_upper"] = float(np.percentile(vals, (1.0 - alpha / 2.0) * 100))
                        results_stats[section] = stats_result
        
        return results_stats

    def perform_bootstrapping(self, predictions, labels, probabilities):
        """Run bootstrapping and merger metrics."""
        logger.info(f"Running {self.n_bootstraps} bootstrap iterations...")
        
        n_samples = len(predictions)
        bootstrap_metrics = []
        
        
        attempts = 0
        max_attempts = self.n_bootstraps * 10
        
        pbar = tqdm(total=self.n_bootstraps, desc="Bootstrapping")
        
        while len(bootstrap_metrics) < self.n_bootstraps and attempts < max_attempts:
            # Resample indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Get resampled data
            res_preds = predictions[indices]
            res_labels = labels[indices]
            res_probs = probabilities[indices] if probabilities is not None else None
            
            # Evaluate
            if "classification" in self.task_type:
                res = self.evaluate_classification(res_preds, res_labels, res_probs)
            elif "survival" in self.task_type:
                res = self.evaluate_survival(res_preds, res_labels)
            
            
            # Check validity of results (e.g. check if at least one c-index was calculated)
            # A sample is valid if we have at least one valid metric for classification or survival
            is_valid = False
            
            if "classification" in self.task_type:
                 # Check if we have valid macro metrics
                 if res.get("macro_averaged_metrics") and res["macro_averaged_metrics"].get("auc") is not None: # AUC usually
                      is_valid = True
                 # Or check if we have any valid per-label metric
                 elif any(m.get("auroc") is not None for m in res.get("per_label_metrics", {}).values()):
                      is_valid = True
                 # If no AUC, maybe other metrics are valid, usually classification always yields something unless empty
                 else:
                      is_valid = True # Assume valid unless empty
                      
            elif "survival" in self.task_type:
                 # Check if we have valid C-Index
                 # Check macro C-index
                 macro_c = res.get("macro_averaged_metrics", {}).get("c_index")
                 if macro_c is not None:
                      is_valid = True
                 else:
                      # Check if ANY label has a valid c-index
                      for label_res in res.get("per_label_metrics", {}).values():
                           if label_res.get("c_index") is not None:
                                is_valid = True
                                break
            
            if is_valid:
                bootstrap_metrics.append(res)
                # Update progress bar
                pbar.update(1)
            else:
                logger.warning(f"Bootstrap sample {attempts} failed to produce valid metrics. Retrying...")
            
            attempts += 1
            
        pbar.close()
            
        if len(bootstrap_metrics) < self.n_bootstraps:
             logger.warning(f"Could only collect {len(bootstrap_metrics)} valid bootstrap samples out of {attempts} attempts.")
        
        # if self.save_predictions and len(bootstrap_metrics) > 0:
        #     bootstrap_predictions_dir = self.output_dir / "bootstrap_predictions"
        #     bootstrap_predictions_dir.mkdir(exist_ok=True)
            
        #     for i, res in enumerate(bootstrap_metrics):
        #         bootstrap_pred_path = bootstrap_predictions_dir / f"bootstrap_pred_{i+1}.pt"
        #         torch.save({
        #             "predictions": res["predictions"],
        #             "probabilities": res["probabilities"],
        #             "labels": res["labels"],
        #             "ages": res["ages"]
        #         }, bootstrap_pred_path)
        #         logger.info(f"Saved bootstrap predictions {i+1}/{len(bootstrap_metrics)} to {bootstrap_pred_path}")

        # Compute stats
        stats = self.compute_confidence_intervals(bootstrap_metrics)
        return stats

    def run(self):
        """Run the complete evaluation pipeline."""
        logger.info("="*80)
        logger.info("Starting Model Evaluation")
        logger.info("="*80)

        # Setup dataset
        self.setup_dataset()
        
        # Load model
        self.load_checkpoint_and_config()
        
        # Get dataloader
        dataloader = self.get_dataloader()
        
        # Collect predictions
        predictions, labels, probabilities, ages, ids = self.collect_predictions(dataloader)
        
        # Calculate metrics based on task type
        if "classification" in self.task_type:
            results = self.evaluate_classification(predictions, labels, probabilities)
        elif "survival" in self.task_type:
            results = self.evaluate_survival(predictions, labels)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        # Perform bootstrapping if requested
        if self.n_bootstraps > 0:
            bootstrap_stats = self.perform_bootstrapping(predictions, labels, probabilities)
            
            # Merge stats into results
            # We want to add the mean/ci fields to the existing dicts
            
            # Per label
            if "per_label_metrics" in bootstrap_stats:
                for label, stats in bootstrap_stats["per_label_metrics"].items():
                    if label in results["per_label_metrics"]:
                        results["per_label_metrics"][label].update(stats)
            
            # Macro/Micro
            for section in ["macro_averaged_metrics", "micro_averaged_metrics"]:
                if section in bootstrap_stats:
                     if section not in results:
                         results[section] = {}
                     results[section].update(bootstrap_stats[section])
        
        if self.age_metrics:
            if self.age_metrics == "auto":
                # Compute quartiles for age stratification
                self.age_metrics = [int(np.percentile(ages, p)) for p in [25, 50, 75]]
                logger.info(f"Auto-calculated age thresholds: {self.age_metrics}")
            
            # Sort thresholds and define ranges
            age_ths = sorted(list(set(self.age_metrics)))
            age_ranges = []
            
            # First range: min to first threshold
            age_ranges.append((int(min(ages)), age_ths[0]))
            
            # Middle ranges
            for i in range(len(age_ths) - 1):
                age_ranges.append((age_ths[i], age_ths[i+1]))
            
            # Last range: last threshold + 1 to max+1
            if age_ths[-1] < max(ages):
                age_ranges.append((age_ths[-1], int(max(ages))+1))
            
            for min_age_r, max_age_r in age_ranges:
                logger.info(f"Evaluating for age range: {min_age_r} - {max_age_r+1}")
                indx_age = (ages >= min_age_r) & (ages <= max_age_r)
                
                if not np.any(indx_age):
                    logger.warning(f"No samples found for age range {min_age_r}-{max_age_r+1}. Skipping.")
                    continue
                    
                predictions_age = predictions[indx_age]
                labels_age = labels[indx_age]
                probabilities_age = probabilities[indx_age] if probabilities is not None else None
                
                bootstrap_stats_ages = self.perform_bootstrapping(predictions_age, labels_age, probabilities_age)
                
                age_key = f'age_{min_age_r}_{max_age_r+1}' if max_age_r == int(max(ages)) else f'age_{min_age_r}_{max_age_r}'
                
                # Per label
                if "per_label_metrics" in bootstrap_stats_ages:
                    for label, stats in bootstrap_stats_ages["per_label_metrics"].items():
                        if label in results["per_label_metrics"]:
                            if age_key not in results["per_label_metrics"][label]:
                                results["per_label_metrics"][label][age_key] = {}
                            results["per_label_metrics"][label][age_key].update(stats)
                
                # Macro/Micro
                for section in ["macro_averaged_metrics", "micro_averaged_metrics"]:
                    if section in bootstrap_stats_ages:
                        if section not in results:
                            results[section] = {}
                        if age_key not in results[section]:
                            results[section][age_key] = {}
                        results[section][age_key].update(bootstrap_stats_ages[section])


        # Add metadata
        results["checkpoint_path"] = str(self.checkpoint_path)
        results["dataset_split"] = self.dataset_split
        results["label_names"] = self.label_names
        results["n_bootstraps"] = self.n_bootstraps
        results["ci_level"] = self.ci_level
        
        # Save results
        self.save_results(results)
        
        # Optionally save predictions
        if self.save_predictions:
            self.save_predictions_to_file(predictions, labels, probabilities, ages, ids)
        
        # Print summary
        self.print_summary(results)
        
        logger.info("="*80)
        logger.info("Evaluation Complete")
        logger.info("="*80)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of evaluation results."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Task Type: {results['task_type']}")
        logger.info(f"Dataset Split: {results['dataset_split']}")
        logger.info(f"Number of Samples: {results['num_samples']}")
        logger.info(f"Number of Labels: {results['num_labels']}")
        logger.info("")
        
        # Per-label results
        logger.info("PER-LABEL METRICS:")
        logger.info("-"*80)
        for label_name, metrics in results["per_label_metrics"].items():
            logger.info(f"\n{label_name}:")
            for metric_name, value in metrics.items():
                if value is not None:
                    if isinstance(value, dict):
                        logger.info(f"  {metric_name}: {value}")
                    elif isinstance(value, float):
                        logger.info(f"  {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: {value}")
        
        # Averaged results
        logger.info("\n" + "-"*80)
        logger.info("AVERAGED METRICS:")
        logger.info("-"*80)
        
        for avg_type in ["macro_averaged_metrics", "micro_averaged_metrics"]:
            if avg_type in results and results[avg_type]:
                logger.info(f"\n{avg_type.replace('_', ' ').title()}:")
                for metric_name, value in results[avg_type].items():
                    if value is not None:
                        if isinstance(value, dict):
                            logger.info(f"  {metric_name}: {value}")
                        elif isinstance(value, float):
                            logger.info(f"  {metric_name}: {value:.4f}")
                        else:
                            logger.info(f"  {metric_name}: {value}")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on specified dataset"
    )
    
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        required=True,
        help="Path to model checkpoint folder "
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--only_demo_model",
        action="store_true",
        help="Only evaluate the demographic model"
    )

    parser.add_argument(
        "--task_config",
        type=str,
        required=True,
        help="Path to task configuration YAML file"
    )

    parser.add_argument(
        "--outcome_csv_file_path",
        type=str,
        required=True,
        help="Path to outcome CSV file"
    )
    
    parser.add_argument(
        "--demographic_csv_file_path",
        type=str,
        required=True,
        help="Path to demographic CSV file"
    )

    parser.add_argument(
        "--dataset_split_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Which dataset split to evaluate on (default: test)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save model predictions to file"
    )
    
    parser.add_argument(
        "--auroc_years",
        type=float,
        nargs="+",
        default=None,
        help="Time points in years for time-dependent AUROC calculation (survival tasks only). Example: --auroc_years 1 3 5 10"
    )

    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=0,
        help="Number of bootstrap samples for confidence intervals (default: 0, no bootstrapping)"
    )

    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence interval level (default: 0.95)"
    )

    parser.add_argument(
        "--strat_ages_metrics",
        action="store_true",
        help="calculate metrics per ages quartiles"
    )

    parser.add_argument(
        "--ages_values_metrics",
        type=int,
        nargs="+",
        default=None,
        help="ages threshold for calculate metrics"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.task_config, 'r') as f:
        task_config = yaml.safe_load(f)

    config = update_dict(config, task_config)

    config['training_config']['split_path'] = args.dataset_split_path
    config['csv_config']['original_csv_file'] = args.outcome_csv_file_path
    config['csv_config']['demographic_csv_file'] = args.demographic_csv_file_path

    chekpoint_files = [file for file in os.listdir(args.checkpoint_folder) if file.endswith(".ckpt")]
    best_checkpoint = [file for file in chekpoint_files if 'best' in file][0]
    checkpoint_path = os.path.join(args.checkpoint_folder, best_checkpoint)

    if args.ages_values_metrics:
        ages_th = args.ages_values_metrics
    elif args.strat_ages_metrics:
        # Defer to run() — percentiles will be computed from event-subject ages
        ages_th = "auto"
    else:
        ages_th = None
    
    # Create evaluator
    evaluator = ModelEvaluator(
        checkpoint_path=checkpoint_path,
        config=config,
        only_demo=args.only_demo_model,
        dataset_split=args.dataset,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        auroc_time_points=args.auroc_years,
        n_bootstraps=args.n_bootstraps,
        ci_level=args.ci_level,
        age_metrics = ages_th
    )
    
    # Run evaluation
    results = evaluator.run()
    
    return results


if __name__ == "__main__":
    main()
