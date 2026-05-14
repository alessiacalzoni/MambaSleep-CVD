import json
import os
import wandb
import yaml
import argparse
import pytorch_lightning as pl
import logging
import torch
import sys
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from torchmetrics import MetricCollection
from pytorch_lightning.loggers import WandbLogger

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.sources.base_system.loss import SurvivalLoss, StratifiedSurvivalLoss
from scripts.sources.base_system.metric import SurvivalMetric
from scripts.sources.dataset_raw import RawMultimodalDataModule, StratifiedRawMultimodalDataModule
from scripts.sources.utils.utils import count_parameters, update_dict, merge_sweep_config
from scripts.sources.models.models_raw import LitRawMultimodalModel
from scripts.sources.models.models import LitDemoBaselineModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def from_path_to_ids(paths):
    return [int(Path(path).stem.split('-')[1]) for path in paths]

def run_trainer(base_params, use_wandb=False, folder_checkpoint=None, only_demo_model=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    pl.seed_everything(42, workers=True)

    # Bypass Docker /dev/shm shared memory limit to prevent Bus Errors
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    # --- WandB Initialization ---
    if use_wandb:
        group_name = base_params.get('wandb_config', {}).get('group_name', "")

        wandb.init(
            project=base_params.get('wandb_config', {}).get('project_name', 'PSG_cardiovascular_survival_raw'),
            group=group_name,
            config=base_params,
        )
        
        config = merge_sweep_config(base_params, wandb.config)
        
        wandb_logger = WandbLogger(
            project=config['wandb_config'].get('project_name', 'PSG_cardiovascular_survival_raw'),
            group=group_name,
            log_model=True
        )
    else:
        config = base_params
        group_name = base_params.get('wandb_config', {}).get('group_name', "")
    
    logger.info(f"Final Configuration being used:\n {json.dumps(config, indent=4, default=str)}")

    csv_config = config['csv_config'].copy()
    labels_to_use = csv_config.get('target_outcome', None)

    if labels_to_use and config.get('file_naming', 'shhs') != 'shhs':
        original_csv = csv_config['original_csv_file']
        df = pd.read_csv(original_csv)

        # Build list of columns to keep: id + selected labels + their _time counterparts
        keep_cols = ['id']
        available_cols = [c for c in df.columns if c != 'id']
        for lbl in labels_to_use:
            # Case-insensitive match
            match = [c for c in available_cols if c.lower() == lbl.lower() and '_time' not in c.lower()]
            if match:
                keep_cols.append(match[0])
                time_col = match[0] + '_time'
                if time_col in df.columns:
                    keep_cols.append(time_col)
            else:
                logger.warning(f"Label '{lbl}' not found in CSV columns: {available_cols}")

        df_filtered = df[keep_cols]
        # Write filtered CSV next to original
        filtered_csv = original_csv.replace('.csv', f'_filtered_{len(labels_to_use)}labels.csv')
        df_filtered.to_csv(filtered_csv, index=False)
        csv_config['original_csv_file'] = filtered_csv
        logger.info(f"Filtered CSV ({len(labels_to_use)} labels) written to: {filtered_csv}")
        logger.info(f"Labels selected: {[c for c in keep_cols if c != 'id' and '_time' not in c]}")

    # --- Data Split Loading ---
    test_ids = None
    val_ids = None

    if isinstance(config['training_config'].get('split_path', None), str):
        with open(config['training_config'].get('split_path'), 'r') as f:
            splits = json.load(f)
            if 'test' in splits.keys():
                test_ids = splits['test']
                if len(test_ids) > 0 and isinstance(test_ids[0], str):
                    test_ids = from_path_to_ids(test_ids)
                    splits['test'] = test_ids
            else:
                test_ids = None
            if 'val' in splits.keys():
                val_ids = splits['val']
                if len(val_ids) > 0 and isinstance(val_ids[0], str):
                    val_ids = from_path_to_ids(val_ids)
                    splits['val'] = val_ids
            else:
                val_ids = None
        
        split_to_dump = {}
        needs_dump = False
        for key in splits.keys():
            if len(splits[key]) > 0 and isinstance(splits[key][0], str):
                splits[key] = from_path_to_ids(splits[key])
                needs_dump = True
            split_to_dump[key] = splits[key]
        if needs_dump:
            with open(config['training_config'].get('split_path'), 'w') as f:
                    json.dump(split_to_dump, f, indent=4)

    # --- DataModule (handles splitting, scaling, dataset creation, dataloaders) ---
    if config.get('use_stratified', False):
        dataset = StratifiedRawMultimodalDataModule(
            n_bins=config.get('n_bins', 10),
            config_csv=csv_config,
            data_folder=config['paths'].get('data_folder'),
            model_config=config['model_config'],
            batch_size=config['training_config'].get('batch_size'),
            num_workers=config['training_config'].get('num_workers'),
            test=test_ids if test_ids is not None else config['training_config'].get('test', 0.2),
            val=val_ids if val_ids is not None else config['training_config'].get('val', 0.2),
            random_state=config['training_config'].get('random_state'),
            save_splits=config['training_config'].get('save_splits'),
            file_naming=config.get('file_naming', 'shhs'),
        )
    else:
        dataset = RawMultimodalDataModule(
            config_csv=csv_config,
            data_folder=config['paths'].get('data_folder'),
            model_config=config['model_config'],
            batch_size=config['training_config'].get('batch_size'),
            num_workers=config['training_config'].get('num_workers'),
            test=test_ids if test_ids is not None else config['training_config'].get('test', 0.2),
            val=val_ids if val_ids is not None else config['training_config'].get('val', 0.2),
            random_state=config['training_config'].get('random_state'),
            save_splits=config['training_config'].get('save_splits'),
            file_naming=config.get('file_naming', 'shhs'),
        )

    logger.info(f"Total outcome distribution: {dataset.label_distribution()}")
    logger.info(f"Outcome distribution in train: {dataset.train_dataset.label_distribution()}")
    logger.info(f"Outcome distribution in val: {dataset.val_dataset.label_distribution()}")
    logger.info(f"Outcome distribution in test: {dataset.test_dataset.label_distribution()}")

    n_labels = dataset.n_labels

    # --- Pos-Weight Calculation ---
    pos_weight = None
    factor = config['training_config'].get('pos_weight_factor', None)
    
    if factor:
        if n_labels == 1:
            target_col = dataset.labels_names[0]
            counts = dataset.train_dataframe[target_col].value_counts()
            num_neg = counts.get(0, 0)
            num_pos = counts.get(1, 1)
            weight_value = (num_neg / num_pos) * factor
            pos_weight = torch.tensor(weight_value, dtype=torch.float32)
            logger.info(f"Using binary pos_weight: {weight_value:.4f} (Base: {num_neg/num_pos:.2f}, Factor: {factor})")
        elif n_labels > 1:
            weights_list = []
            logger.info(f"Calculating pos_weights for {n_labels} labels with factor {factor}...")
            for col in dataset.labels_names:
                counts = dataset.train_dataframe[col].value_counts()
                num_neg = counts.get(0, 0)
                num_pos = counts.get(1, 1)
                w = (num_neg / num_pos) * factor
                weights_list.append(w)
                logger.info(f"  Label '{col}': weight {w:.4f} (Base: {num_neg/num_pos})")
            pos_weight = torch.tensor(weights_list, dtype=torch.float32)

    # --- Loss & Metrics ---
    if config.get('use_stratified', False):
        loss = StratifiedSurvivalLoss(min_events=config.get('min_events', 3), pos_weight=pos_weight)
    else:
        loss = SurvivalLoss(pos_weight=pos_weight)

    metrics = MetricCollection({
        'C_index_macro': SurvivalMetric(input_args=['hazards', 'is_event', 'event_time'], average='macro'),
        'C_index_micro': SurvivalMetric(input_args=['hazards', 'is_event', 'event_time'], average='micro'),
    })

    # --- Model ---
    if only_demo_model:
        lit_model = LitDemoBaselineModel(
            n_demo_features = len(config['csv_config']['demographic_stratification_var'])+1 if 'med' in config['csv_config']['demographic_stratification_var'] else len(config['csv_config']['demographic_stratification_var']),
            model_config=config['model_config'],
            number_labels=n_labels,
            lr=config['training_config'].get('lr'),
            weight_decay=config['training_config'].get('weight_decay'),
            loss=loss,
            train_metrics=metrics,
            val_metrics=metrics,
            test_metrics=metrics,
        )
    else:
        lit_model = LitRawMultimodalModel(
            model_config=config['model_config'],
            number_labels=n_labels,
            lr=config['training_config'].get('lr'),
            weight_decay=config['training_config'].get('weight_decay'),
            loss=loss,
            train_metrics=metrics,
            val_metrics=metrics,
            test_metrics=metrics,
        )

        coupled_mamba_type = type(lit_model.model.coupled_mamba).__name__
        logger.info(f"\n{'='*50}\nACTIVE MODEL CLASS: {coupled_mamba_type}\n{'='*50}\n")
        
        if use_wandb:
            try:
                wandb.config.update({"actual_coupled_mamba_class": coupled_mamba_type})
            except Exception:
                pass
                
    logger.info("Model structure:\n")
    logger.info(lit_model.model)

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=config['training_config'].get('epochs'),
        log_every_n_steps=5,
        gradient_clip_val=1,
        accumulate_grad_batches=config['training_config'].get('accumulate_grad_batches'),
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True,
        enable_checkpointing=True,
        inference_mode=False,
        logger=wandb_logger if use_wandb else TensorBoardLogger(
            save_dir=os.path.join(config['paths'].get('checkpoint'), 'survival', 'stratified' if config.get('use_stratified', False) else "", 'raw', f'{folder_checkpoint}_only_demo' if only_demo_model else f'{folder_checkpoint}')
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=True,
                dirpath=os.path.join(
                    config['paths'].get('checkpoint'), group_name,
                    wandb.run.name) if wandb.run else os.path.join(config['paths'].get('checkpoint'), 'survival', 'stratified' if config.get('use_stratified', False) else "", 'raw', f'{folder_checkpoint}_only_demo' if only_demo_model else f'{folder_checkpoint}'),
                filename='best-checkpoint_epoch={epoch}_val-loss={val_loss:.4f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=config['training_config'].get('patience'),
            ),
        ]
    )

    # --- Train & Test ---
    try:
        trainer.fit(lit_model, dataset.train_dataloader(), dataset.val_dataloader())
        logger.info(f"Calculated params after fit: {count_parameters(lit_model)}")
        trainer.test(lit_model, dataset.test_dataloader(), ckpt_path='best')

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
        
    wandb.finish() if use_wandb else None
    logger.info("Training finished.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train raw signal multimodal survival model")
    parser.add_argument("--global_config", type=str, help="Path to the global configuration file", required=True)
    parser.add_argument("--task_config", type=str, help="Path to the task configuration file", required=True)
    parser.add_argument("--only_demo_model", action="store_true", help="Only demo model", required=False)
    parser.add_argument("--use_wandb", action="store_true")
    
    # Options for stratified & masking
    parser.add_argument("--use_stratified", action="store_true", help="Use stratified dataloader and loss")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of age bins for stratification")
    parser.add_argument("--min_events", type=int, default=3, help="Minimum events per stratum")
    parser.add_argument("--imputing_method", type=str, choices=["remove_events", "censor_date", "mask_events"], default="censor_date")
    
    args = parser.parse_args()

    with open(args.global_config) as global_config:
        params = yaml.safe_load(global_config)

    with open(args.task_config, "r") as config_task:
        task_params = yaml.safe_load(config_task)

    params = update_dict(params, task_params)
    
    params['use_stratified'] = args.use_stratified
    params['n_bins'] = args.n_bins
    params['min_events'] = args.min_events
    if 'csv_config' not in params:
        params['csv_config'] = {}
    params['csv_config']['imputing_method'] = args.imputing_method

    run_trainer(params, args.use_wandb, Path(args.global_config).stem, args.only_demo_model)
