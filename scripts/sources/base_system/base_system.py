import pytorch_lightning as pl
import torch.optim as optim
from typing import List
from torchmetrics import MetricCollection, Metric
import re
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSystem(pl.LightningModule):
    def __init__(
        self, loss : object = None,
        lr: float = None,
        train_metrics : MetricCollection = None,
        val_metrics : MetricCollection = None,
        test_metrics : MetricCollection = None,
        log_filter : List[str] = None,
        log_filter_whitelist : bool = False
    ):
        super().__init__()

        self.__check_loss(loss)

        self.train_metrics = train_metrics.clone(prefix='train_') if train_metrics else None
        self.valid_metrics = val_metrics.clone(prefix='val_') if val_metrics else None
        self.test_metrics = test_metrics.clone(prefix='test_') if test_metrics else None

        self.__check_metrics_input(train_metrics, prefix="(train)")
        self.__check_metrics_input(val_metrics, prefix="(val)")
        self.__check_metrics_input(test_metrics, prefix="(test)")

        self.loss = loss
        self.lr = lr
        self.log_filter = log_filter
        self.log_filter_whitelist = log_filter_whitelist
        # # ---------------------------------------------------------
        # # SANITY CHECK MODE: ON
        # # ---------------------------------------------------------
        # # 1. Fake inputs (Random)
        # self.fake_x = {'eeg': 100*torch.randn(64, 2, 27000, 6).to('cuda') , 'eog': 100*torch.randn(64, 2, 27000, 6).to('cuda')}
        # self.fake_mask = {'eeg': torch.ones(64, 27000).to('cuda') , 'eog': torch.ones(64, 27000).to('cuda')}
        # # 2. Fake labels (Balanced)
        # self.fake_y = torch.randint(0, 2, (64, 1)).to('cuda')
        
    def __check_loss(self, loss):
        logger.info(loss.input_args)
        for loss_arg in loss.input_args:
            if loss_arg not in self.step_args:
                raise KeyError(f"Loss of type '{type(loss).__name__}' requires input_arg {loss_arg} but it is not defined as a return argument of the current system")

    def __check_metrics_input(self, metrics, prefix=""):
        k_map = {
            '(train)': self.train_metrics,
            '(val)': self.valid_metrics,
            '(test)': self.test_metrics
        }

        if not k_map[prefix]:
            return

        for metric_name in metrics.keys():
            metric = metrics[metric_name]
            for metric_arg in metric.input_args:
                if metric_arg not in self.step_args:
                    raise KeyError(f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires input_arg {metric_arg} but it is not defined as a returned argument of the current system")

    def __retrieve_loss_batch_args(self, batch, step_dict):
        for loss_arg in self.loss.batch_args:
            if loss_arg not in batch.keys():
                raise KeyError(f"Loss '{type(self.loss).__name__}' requires batch_arg {loss_arg} but it is not found in the given batch")
            step_dict[loss_arg] = batch[loss_arg]
        return step_dict

    def __retrieve_metrics_batch_args(self, metrics, batch, step_dict, prefix=""):
        if not metrics:
            return step_dict

        for metric_name in metrics.keys():
            metric = metrics[metric_name]
            for metric_arg in metric.batch_args:
                if metric_arg not in batch.keys():
                    raise KeyError(f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires batch_arg {metric_arg} but it is not found in the given batch")
                step_dict[metric_arg] = batch[metric_arg]
        return step_dict

    def __apply_log_filter(self, log_dict):
        if self.log_filter:
            res = {}
            for filt in self.log_filter:
                for k, v in log_dict.items():
                    does_match = bool(re.search(filt, k))

                    if not (does_match ^ self.log_filter_whitelist):
                        res[k] = v
            return res
        else:
            return log_dict

    def configure_optimizers(self):
        """ TODO make configureable """

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, total_steps = self.trainer.estimated_stepping_batches, pct_start = 0.1, anneal_strategy = 'cos')
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}

    @property
    def step_args(self):
        raise NotImplementedError("Make sure to define the step_args property in each system")

    def _step(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        if self.train_metrics:
            # Detach tensors to avoid storing computation graph in metrics which causes deepcopy errors
            metric_step_dict = {
                k: v.detach() if isinstance(v, torch.Tensor) else v 
                for k, v in step_dict.items()
            }
            metric_step_dict = self.__retrieve_metrics_batch_args(self.train_metrics, batch, metric_step_dict, prefix="(train)")
            self.train_metrics.update(metric_step_dict)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        # # ---------------------------------------------------------
        # # SANITY CHECK MODE: ON
        # # Ignoriamo il batch reale e ne creiamo uno finto facile
        # # ---------------------------------------------------------
        # batch = (self.fake_x, self.fake_mask, self.fake_y)
        # # 3. Forward pass con i dati finti
        # step_dict = self._step(batch)  # Il tuo modello processa il rumore random
        # step_dict = self.__retrieve_loss_batch_args(batch, step_dict)
        
        # # 4. Calcolo Loss Semplice (NO Focal Loss, NO pos_weight)
        # # Usiamo la loss standard per vedere se il meccanismo funziona
        # sanity_criterion = torch.nn.BCEWithLogitsLoss()
        # loss = sanity_criterion(step_dict['logits'], self.fake_y.float())

        # # 5. Logga per vedere cosa succede
        # # Calcoliamo una accuracy veloce
        # preds = (torch.sigmoid(step_dict['logits']) > 0.5).float()
        # acc = (preds == self.fake_y).float().mean()
        
        # self.log("sanity_loss", loss, prog_bar=True)
        # self.log("sanity_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        if self.valid_metrics:
            metric_step_dict = {
                k: v.detach() if isinstance(v, torch.Tensor) else v 
                for k, v in step_dict.items()
            }
            metric_step_dict = self.__retrieve_metrics_batch_args(self.valid_metrics, batch, metric_step_dict, prefix="(val)")
            self.valid_metrics.update(metric_step_dict)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return step_dict

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self.__retrieve_loss_batch_args(batch, step_dict)
        
        loss, logs = self.loss(step_dict)
        logs = self.__apply_log_filter(logs)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})

        if self.test_metrics:
            metric_step_dict = {
                k: v.detach() if isinstance(v, torch.Tensor) else v 
                for k, v in step_dict.items()
            }
            metric_step_dict = self.__retrieve_metrics_batch_args(self.test_metrics, batch, metric_step_dict, prefix="(test)")
            self.test_metrics.update(metric_step_dict)

        # self.log("test_loss", loss)

        return step_dict

    def predict_step(self, batch, batch_idx):
        return self._step(batch)

    def on_train_epoch_end(self):
        if self.train_metrics:
            metric_results = self.train_metrics.compute()
            self.log_dict(metric_results)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.valid_metrics:
            metric_results = self.valid_metrics.compute()
            self.log_dict(metric_results)
            self.valid_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            metric_results = self.test_metrics.compute()
            self.log_dict(metric_results)
            self.test_metrics.reset()

class BaseMetric(Metric):
    @property
    def batch_args(self):
        return []

    @property
    def input_args(self):
        raise NotImplementedError()

class BaseLoss(pl.LightningModule):
    """
    Base class, used for inheritance later on.

    Attributes:
        reduction: String defining how to reduced the computed loss per sample 
        on the batch level. Options are: 'mean', 'sum' and 'sum_mean'. This last 
        option only works if the input is three dimension, and uses a sum for the 
        second and third dimension, while the a mean over the batch dimension is taken.
    """

    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        assert reduction in ['sum', 'mean', 'sum_mean'], "Please provide one of the following for the reduction function: 'mean', 'sum' of 'sum_mean'."

        self.reduction = reduction
        self.only_log_num = True

    @property
    def input_args(self):
        # list of key names from the model's step method that are required for the loss
        raise NotImplementedError("Make sure to define the input_args property in each loss class")

    @property
    def batch_args(self):
        # list of key names directly from the original batch that are required for the loss
        return []

    def apply_reduction(self, loss, skip_if_none : bool = False):
        """
        Reduces the computed loss per sample on the batch level.

        Args:
            loss: The computed loss per sample.
            skip_if_none: TODO
        """

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum_mean':
            if len(loss.shape) != 3:
                logger.info('problems with shape')
            return loss.sum(dim=(1, 2)).mean()
        # TODO, must be a better way to handle this
        # skip_if_none is generally controlled by the only_log_num attribute to make sure no N-dimensional tensors are logged
        return loss if not skip_if_none else None