import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.base_system.loss import SurvivalLoss
from sources.dataset import MultimodalDataModule
from sources.models.models import LitDemoBaselineModel, LitMultimodalModel
from sources.models.models_raw import LitRawMultimodalModel
from sources.utils.utils import update_dict

try:
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.cindex import ConcordanceIndex

    TORCHSURV_AVAILABLE = True
except ImportError:
    TORCHSURV_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def from_path_to_ids(paths):
    return [int(Path(path).stem.split("-")[1]) for path in paths]


class ModelEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        mode: str,
        only_demo: bool = False,
        dataset_split: str = "test",
        output_dir: str = "evaluation_results",
        save_predictions: bool = False,
        auroc_time_points: Optional[List[float]] = None,
        n_bootstraps: int = 0,
        ci_level: float = 0.95,
    ):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.mode = mode
        self.only_demo = only_demo
        self.dataset_split_name = dataset_split
        if dataset_split == "primary_prev":
            self.dataset_split = "val"
        elif dataset_split == "secondary_prev":
            self.dataset_split = "test"
        else:
            self.dataset_split = dataset_split
        self.output_dir = Path(output_dir)
        self.save_predictions = save_predictions
        self.auroc_time_points = auroc_time_points
        self.n_bootstraps = n_bootstraps
        self.ci_level = ci_level

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_dataset(self):
        test_ids, val_ids = None, None
        split_path = self.config["training_config"].get("split_path")
        if isinstance(split_path, str):
            with open(split_path, "r") as f:
                self.output_dir = Path(os.path.join(self.output_dir, Path(split_path).stem))
                os.makedirs(self.output_dir, exist_ok=True)
                splits = json.load(f)

            if splits.get("test"):
                test_ids = splits["test"]
                if isinstance(test_ids[0], str):
                    test_ids = from_path_to_ids(test_ids)
            if splits.get("val"):
                val_ids = splits["val"]
                if isinstance(val_ids[0], str):
                    val_ids = from_path_to_ids(val_ids)
            if "secondary_prev" in splits:
                test_ids = from_path_to_ids(splits["secondary_prev"]) if isinstance(splits["secondary_prev"][0], str) else splits["secondary_prev"]
                val_ids = from_path_to_ids(splits["primary_prev"]) if isinstance(splits["primary_prev"][0], str) else splits["primary_prev"]

        self.datamodule = MultimodalDataModule(
            config_csv=self.config["csv_config"],
            data_folder=self.config["paths"]["data_folder"],
            model_config=self.config["model_config"],
            input_mode=self.mode,
            batch_size=1,
            num_workers=0,
            test=test_ids if test_ids is not None else self.config["training_config"].get("test", 0.2),
            val=val_ids if val_ids is not None else self.config["training_config"].get("val", 0.2),
            random_state=self.config["training_config"].get("random_state", 42),
        )
        self.label_names = self.datamodule.labels_names

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        loss = SurvivalLoss()
        demo_vars = self.config["csv_config"]["demographic_stratification_var"]
        n_demo = len(demo_vars) + 1 if "med" in demo_vars else len(demo_vars)
        model_kwargs = dict(
            model_config=self.config["model_config"],
            number_labels=self.datamodule.n_labels,
            lr=self.config["training_config"].get("lr"),
            weight_decay=self.config["training_config"].get("weight_decay"),
            loss=loss,
        )

        if self.only_demo:
            self.model = LitDemoBaselineModel(n_demo_features=n_demo, **model_kwargs)
        elif self.mode == "raw":
            self.model = LitRawMultimodalModel(**model_kwargs)
        else:
            self.model = LitMultimodalModel(**model_kwargs)

        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()
        self.model.to(self.device)

        if self.mode == "raw" and torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            logger.info("cuDNN disabled for long-sequence evaluation")

    def get_dataloader(self):
        ds = {"test": self.datamodule.test_dataset, "val": self.datamodule.val_dataset, "train": self.datamodule.train_dataset}[self.dataset_split]
        return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collate_fn)

    def collect_predictions(self, dataloader):
        preds, labels_all, ids_all = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, masks, labels, demo_features, ids = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                if demo_features is not None:
                    demo_features = demo_features.to(self.device)
                outputs = self.model(inputs, masks, demo_features)
                preds.append(outputs.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
                ids_all.extend(ids.cpu().tolist())
        return np.concatenate(preds), np.concatenate(labels_all), np.array(ids_all)

    def evaluate_survival(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        is_event = labels[:, :, 0]
        event_times = labels[:, :, 1]
        results = {
            "num_samples": len(labels),
            "num_labels": labels.shape[1],
            "per_label_metrics": {},
            "macro_averaged_metrics": {}
        }

        for i, name in enumerate(self.label_names):
            hazards = predictions[:, i]
            events = is_event[:, i]
            times = event_times[:, i]
            valid = events >= 0
            hazards, events, times = hazards[valid], events[valid], times[valid]
            num_events = int(events.sum()) if events.size else 0

            metrics = {"c_index": None, "num_events": num_events, "num_censored": int((1 - events).sum()) if events.size else 0}
            if num_events > 0 and TORCHSURV_AVAILABLE:
                try:
                    ht = torch.tensor(hazards, dtype=torch.float32)
                    et = torch.tensor(events, dtype=torch.bool)
                    tt = torch.tensor(times, dtype=torch.float32)
                    cindex_metric = ConcordanceIndex()
                    try:
                        cindex = cindex_metric(estimate=ht, time=tt, event=et)
                        cindex = float(cindex.item())
                    except Exception as e:
                        logger.warning(f"TorchSurv C-Index calculation failed for {label_name}: {e}. Returning None.")
                        cindex = None
                    metrics["c_index"] = cindex
                except Exception:
                    pass

                if self.auroc_time_points:
                    for tp in self.auroc_time_points:
                        try:
                            auc_metric = Auc()
                            auroc = auc_metric(estimate=ht, time=tt, event=et, new_time=int(tp * 365))
                            metrics[f"auroc_{tp}y"] = float(auroc.item())
                        except Exception:
                            metrics[f"auroc_{tp}y"] = None

            results["per_label_metrics"][name] = metrics

        valid_c = [m["c_index"] for m in results["per_label_metrics"].values() if m["c_index"] is not None]
        results["macro_averaged_metrics"]["c_index"] = float(np.mean(valid_c)) if valid_c else None
        
        return results

    def save_results(self, results: Dict):
        name = self.dataset_split_name
        json_path = self.output_dir / f"evaluation_results_{name}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {json_path}")

        rows = [{"label": k, **v} for k, v in results["per_label_metrics"].items()]
        pd.DataFrame(rows).to_csv(self.output_dir / f"evaluation_results_{name}.csv", index=False)
    
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
        
        results_stats = {}
        
        for section in ["per_label_metrics", "macro_averaged_metrics"]:
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
            res = self.evaluate_survival(res_preds, res_labels)
            
            # Check validity of results (e.g. check if at least one c-index was calculated)
            # A sample is valid if we have at least one valid metric for classification or survival
            is_valid = False
                      
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

        stats = self.compute_confidence_intervals(bootstrap_metrics)
        return stats

    def run(self):
        self.setup_dataset()
        self.load_model()
        predictions, labels, ids = self.collect_predictions(self.get_dataloader())
        results = self.evaluate_survival(predictions, labels)
        # Perform bootstrapping if requested
        if self.n_bootstraps > 0:
            bootstrap_stats = self.perform_bootstrapping(predictions, labels, probabilities)
            
            # Per label
            if "per_label_metrics" in bootstrap_stats:
                for label, stats in bootstrap_stats["per_label_metrics"].items():
                    if label in results["per_label_metrics"]:
                        results["per_label_metrics"][label].update(stats)
            
            # Macro
            for section in ["macro_averaged_metrics"]:
                if section in bootstrap_stats:
                     if section not in results:
                         results[section] = {}
                     results[section].update(bootstrap_stats[section])
        
        results.update(
            checkpoint_path=str(self.checkpoint_path),
            dataset_split=self.dataset_split,
            label_names=self.label_names,
            mode=self.mode,
            n_bootstraps = self.n_bootstraps,
            ci_level = self.ci_level
        )
        self.save_results(results)

        if self.save_predictions:
            np.savez(self.output_dir / f"predictions_{self.dataset_split}.npz", predictions=predictions, labels=labels, ids=ids)

        logger.info(f"Macro C-index: {results['macro_averaged_metrics'].get('c_index')}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate survival model (features or raw signals).")
    parser.add_argument("--mode", choices=["features", "raw"], required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task_config", default=None)
    parser.add_argument("--outcome_csv_file_path", required=True)
    parser.add_argument("--demographic_csv_file_path", default=None)
    parser.add_argument("--dataset_split_path", required=True)
    parser.add_argument("--dataset", default="test", choices=["test", "val", "train"])
    parser.add_argument("--output_dir", default="evaluation_results")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--auroc_years", type=float, nargs="+", default=None)
    parser.add_argument("--only_demo_model", action="store_true")
    parser.add_argument("--n_bootstraps", type=int, default=0, help="Number of bootstrap iterations (default: 0 for no bootstrapping)")
    parser.add_argument("--ci_level", type=float, default=0.95, help="Confidence interval level (default: 0.95)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.task_config:
        with open(args.task_config) as f:
            config = update_dict(config, yaml.safe_load(f))

    config["training_config"]["split_path"] = args.dataset_split_path
    config["csv_config"]["original_csv_file"] = args.outcome_csv_file_path
    if args.demographic_csv_file_path:
        config["csv_config"]["demographic_csv_file"] = args.demographic_csv_file_path

    ckpts = [f for f in os.listdir(args.checkpoint_folder) if f.endswith(".ckpt") and "best" in f]
    checkpoint_path = os.path.join(args.checkpoint_folder, ckpts[0])

    evaluator = ModelEvaluator(
        checkpoint_path=checkpoint_path,
        config=config,
        mode=args.mode,
        only_demo=args.only_demo_model,
        dataset_split=args.dataset,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        auroc_time_points=args.auroc_years,
    )
    return evaluator.run()

if __name__ == "__main__":
    main()
