import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np

class ProbabilityDistributionCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.preds = []
        self.targets = []
        self.class_names = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        with torch.no_grad():
            step_dict = pl_module._step(batch)

        probs = torch.sigmoid(step_dict['logits'])
        
        self.preds.append(probs.detach().cpu())
        self.targets.append(step_dict['labels'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        
        all_preds = torch.cat(self.preds).numpy()
        all_targets = torch.cat(self.targets).numpy()
        
        # Clear lists for next epoch
        self.preds.clear()
        self.targets.clear()

        num_classes = all_preds.shape[1]
        
        fig, axes = plt.subplots(nrows = num_classes, ncols = 1, figsize=(10, 6))
        if num_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(num_classes):
            ax = axes[i]
            
            class_probs = all_preds[:, i]  
            class_targets = all_targets[:, i]
            
            label_name = self.class_names[i] if self.class_names else f"Class {i}"
            
            if len(class_probs[class_targets==0]) > 0:
                ax.hist(class_probs[class_targets==0].flatten(), color="red", label="Neg (0)", alpha=0.5, bins=20)
            if len(class_probs[class_targets==1]) > 0:
                ax.hist(class_probs[class_targets==1].flatten(), color="green", label="Pos (1)", alpha=0.5, bins=20)
            
            ax.set_title(f"{label_name}")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Prob")
            if i == 0: ax.legend()

        for j in range(num_classes, len(axes)):
            axes[j].axis('off')

        # Log on WandB
        if trainer.logger:
            trainer.logger.experiment.log({
                "val/probability_distributions": wandb.Image(fig),
                "epoch": trainer.current_epoch
            })
        
        plt.close(fig)