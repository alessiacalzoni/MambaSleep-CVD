import argparse
import json
import logging
import os
import sys
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchmetrics import MetricCollection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.base_system.loss import SurvivalLoss
from sources.base_system.metric import SurvivalMetric
from sources.dataset import MultimodalDataModule
from sources.models.models import LitDemoBaselineModel, LitMultimodalModel
from sources.models.models_raw import LitRawMultimodalModel
from sources.utils.utils import count_parameters, merge_sweep_config, update_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def from_path_to_ids(paths):
    return [int(Path(path).stem.split("-")[1]) for path in paths]


def _load_split_ids(config: dict):
    test_ids, val_ids = None, None
    split_path = config.get("training_config", {}).get("split_path")
    if not isinstance(split_path, str):
        return test_ids, val_ids

    with open(split_path, "r") as f:
        splits = json.load(f)

    if "test" in splits and splits["test"]:
        test_ids = splits["test"]
        if isinstance(test_ids[0], str):
            test_ids = from_path_to_ids(test_ids)
            splits["test"] = test_ids

    if "val" in splits and splits["val"]:
        val_ids = splits["val"]
        if isinstance(val_ids[0], str):
            val_ids = from_path_to_ids(val_ids)
            splits["val"] = val_ids

    to_dump = {}
    needs_dump = False
    for key, value in splits.items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            splits[key] = from_path_to_ids(value)
            needs_dump = True
        to_dump[key] = splits[key]
    if needs_dump:
        with open(split_path, "w") as f:
            json.dump(to_dump, f, indent=4)

    return test_ids, val_ids


def _init_wandb(config: dict, use_wandb: bool, default_project: str):
    if not use_wandb:
        return config, None, config.get("wandb_config", {}).get("group_name", "")

    import wandb

    group_name = config.get("wandb_config", {}).get("group_name", "")
    wandb.init(
        project=config.get("wandb_config", {}).get("project_name", default_project),
        group=group_name,
        config=config,
    )
    merged = merge_sweep_config(config, wandb.config)
    wandb_logger = WandbLogger(
        project=merged.get("wandb_config", {}).get("project_name", default_project),
        group=group_name,
        log_model=True,
    )
    return merged, wandb_logger, group_name


def run_trainer(base_params: dict, mode: str, use_wandb: bool = False, folder_checkpoint: str = None, only_demo_model: bool = False):
    is_raw = mode == "raw"
    default_project = "PSG_cardiovascular_survival_raw" if is_raw else "PSG_cardiovascular_survival"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pl.seed_everything(42, workers=True)

    if is_raw:
        torch.multiprocessing.set_sharing_strategy("file_descriptor")

    config, wandb_logger, group_name = _init_wandb(base_params, use_wandb, default_project)
    logger.info(f"Final configuration:\n{json.dumps(config, indent=4, default=str)}")

    test_ids, val_ids = _load_split_ids(config)

    dataset = MultimodalDataModule(
        config_csv=config["csv_config"],
        data_folder=config["paths"]["data_folder"],
        model_config=config["model_config"],
        input_mode=mode,
        batch_size=config["training_config"].get("batch_size"),
        num_workers=config["training_config"].get("num_workers"),
        test=test_ids if test_ids is not None else config["training_config"].get("test", 0.2),
        val=val_ids if val_ids is not None else config["training_config"].get("val", 0.2),
        random_state=config["training_config"].get("random_state"),
        save_splits=config["training_config"].get("save_splits"),
    )

    logger.info(f"Total outcome distribution: {dataset.label_distribution()}")
    logger.info(f"Train: {dataset.train_dataset.label_distribution()}")
    logger.info(f"Val: {dataset.val_dataset.label_distribution()}")
    logger.info(f"Test: {dataset.test_dataset.label_distribution()}")

    n_labels = dataset.n_labels
    pos_weight = None
    factor = config["training_config"].get("pos_weight_factor")
    if factor:
        if n_labels == 1:
            col = dataset.labels_names[0]
            counts = dataset.train_dataframe[col].value_counts()
            weight = (counts.get(0, 0) / counts.get(1, 1)) * factor
            pos_weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weights = []
            for col in dataset.labels_names:
                counts = dataset.train_dataframe[col].value_counts()
                weights.append((counts.get(0, 0) / counts.get(1, 1)) * factor)
            pos_weight = torch.tensor(weights, dtype=torch.float32)

    loss = SurvivalLoss(pos_weight=pos_weight)
    metrics = MetricCollection(
        {
            "C_index_macro": SurvivalMetric(input_args=["hazards", "is_event", "event_time"], average="macro"),
            "C_index_micro": SurvivalMetric(input_args=["hazards", "is_event", "event_time"], average="micro"),
        }
    )

    demo_vars = config["csv_config"]["demographic_stratification_var"]
    n_demo = len(demo_vars) + 1 if "med" in demo_vars else len(demo_vars)
    model_kwargs = dict(
        model_config=config["model_config"],
        number_labels=n_labels,
        lr=config["training_config"].get("lr"),
        weight_decay=config["training_config"].get("weight_decay"),
        loss=loss,
        train_metrics=metrics,
        val_metrics=metrics,
        test_metrics=metrics,
    )

    if only_demo_model:
        lit_model = LitDemoBaselineModel(n_demo_features=n_demo, **model_kwargs)
    elif is_raw:
        lit_model = LitRawMultimodalModel(**model_kwargs)
    else:
        lit_model = LitMultimodalModel(**model_kwargs)

    ckpt_root = config["paths"]["checkpoint"]
    suffix = f"{folder_checkpoint}_only_demo" if only_demo_model else folder_checkpoint
    if is_raw:
        tb_dir = os.path.join(ckpt_root, "raw", suffix)
    else:
        ecg_res = config["model_config"]["resolution_input_embedding_s"].get("ecg", 1)
        backbone = "DANA" if ecg_res == 5 else "USleep"
        tb_dir = os.path.join(ckpt_root, "features", backbone, suffix)

    if use_wandb and wandb.run:
        import wandb
        ckpt_dir = os.path.join(ckpt_root, group_name, wandb.run.name)  
    else:
        ckpt_dir = tb_dir

    trainer = pl.Trainer(
        max_epochs=config["training_config"].get("epochs"),
        log_every_n_steps=5,
        gradient_clip_val=1,
        accumulate_grad_batches=config["training_config"].get("accumulate_grad_batches"),
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        enable_checkpointing=True,
        inference_mode=not is_raw,
        logger=wandb_logger if use_wandb else TensorBoardLogger(save_dir=tb_dir),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                dirpath=ckpt_dir,
                filename="best-checkpoint_epoch={epoch}_val-loss={val_loss:.4f}",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config["training_config"].get("patience"),
            ),
        ],
    )

    trainer.fit(lit_model, dataset.train_dataloader(), dataset.val_dataloader())
    logger.info(f"Params after fit: {count_parameters(lit_model)}")
    trainer.test(lit_model, dataset.test_dataloader(), ckpt_path="best")

    if use_wandb:
        import wandb

        wandb.finish()

    logger.info("Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Train survival model (features or raw signals).")
    parser.add_argument("--mode", choices=["features", "raw"], required=True)
    parser.add_argument("--global_config", type=str, help="Path to the global configuration file", required=True)
    parser.add_argument("--task_config", type=str, help="Path to the task configuration file", required=True)
    parser.add_argument("--only_demo_model", action="store_true", help="Only demo model")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--imputing_method",
        choices=["remove_events", "censor_date", "mask_events"],
        default="mask_events",
    )
    args = parser.parse_args()

    with open(args.global_config) as f:
        params = yaml.safe_load(f)
    with open(args.task_config) as f:
        params = update_dict(params, yaml.safe_load(f))

    params.setdefault("csv_config", {})
    params["csv_config"]["imputing_method"] = args.imputing_method

    run_trainer(
        params,
        mode=args.mode,
        use_wandb=args.use_wandb,
        folder_checkpoint=Path(args.global_config).stem,
        only_demo_model=args.only_demo_model,
    )

if __name__ == "__main__":
    main()
