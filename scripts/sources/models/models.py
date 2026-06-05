import torch
import torch.nn as nn
import numpy as np
import warnings
from einops import rearrange
from typing import Optional
import sys
import os
from torchmetrics import Accuracy
import torch.nn.functional as F
import re
import logging
import torch.optim as optim
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_system.base_system import BaseSystem
from models.efficient_coupled_mamba import EfficientCoupledMambaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_weights(module):
    if isinstance(module, nn.Linear):
        # Apply Kaiming initialization to Linear layers
        if module.weight.requires_grad:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None and module.bias.requires_grad:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
        logger.info("Initialized Linear layer with Kaiming uniform initialization")
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm: weight=1, bias=0
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
        logger.info("Initialized LayerNorm with weight=1, bias=0")

class ChannelPooling(nn.Module):
    def __init__(self, pooling_type: str = 'avg', att_size: Optional[int] = None):
        super().__init__()
        assert pooling_type.lower() in ['avg', 'att'], "Pooling must be either average (\"avg\") or attention based (\"att\")"
        self.pooling_type = pooling_type.lower()
        
        if self.pooling_type == 'att':
            self.att_size = att_size
            self.W_a = nn.LazyLinear(att_size)
            self.a_e = nn.Parameter(torch.randn(att_size))
            
    def forward(self, x, mask=None):
        # x: (B, C, L, F)
        # mask: (B, L) optional
            
        if self.pooling_type == 'avg':
            # Mean over dim 1 (Channels)
            out = x.mean(dim=1) # (B, L, F)
            
        elif self.pooling_type == 'att':
            # Attention mechanism
            # x: (B, C, L, F)
            # a_t = tanh(W_a x + b_a) -> (B, C, L, A)
            a_t = torch.tanh(self.W_a(x))
            
            # scores = a_t @ a_e
            # (B, C, L, A) dot (A) -> (B, C, L)
            scores = torch.matmul(a_t, self.a_e)
            
            # Softmax over channels (dim 1)
            alpha = F.softmax(scores, dim=1) # (B, C, L)
            
            # Weighted sum
            # out = sum_c alpha_c * x_c
            # alpha: (B, C, L) -> (B, C, L, 1)
            out = (x * alpha.unsqueeze(-1)).sum(dim=1) # (B, L, F)
            
            # Apply mask if provided to guarantee zero output/gradients for padded tokens
            if mask is not None:
                # mask: (B, L) -> (B, L, 1)
                out = out * mask.unsqueeze(-1)
            
        return out

class MultimodalModel(nn.Module):
    def __init__(self, config, number_lables: int):
        """
        config: dict containing:
            - modalities: dict {name: [channels]}
            - resolution_input_embedding_s: dict {name: resolution in s}
            - resolution_output_embedding_s: int. This value determines the length (unique) for the sequences (of different modalities) that will be passed to the Mamba model. Length (in seconds) that correspond to a token
            - pooling_config: str|list[str, int] ('avg' or ['att', att_size])
            - dropout: float (optional, default 0)
            - mamba_config: dict (optional)
            - use_efficient_coupled_mamba: bool (optional, default True) if false use the MambaFusion approach
            - use_demographic_features: bool (optional, default False) if true use demographic features in the final classification layer
        number_lables: int
            Number of labels. If 1 it is considered binary classification, otherwise it is considered multilabel classification
        """
        super().__init__()
        
        self.modalities_config = config['modalities']
        self.resolution_input_embedding_s = config.get('resolution_input_embedding_s')
        self.resolution_output_embedding_s = config.get('resolution_output_embedding_s')
        self.factor_resolution_ecg = int(self.resolution_output_embedding_s/self.resolution_input_embedding_s.get('ecg', 1))
        self.use_demographic_features = config.get('use_demographic_features', False)
        # Support both 'pooling_config'
        pooling_config = config.get('pooling_config')
        if isinstance(pooling_config, str):
            self.pooling_type = pooling_config
            att_size = None
        else:
            self.pooling_type, att_size = pooling_config
        self.channel_poolings = nn.ModuleDict()
        self.modality_names = list(self.modalities_config.keys()) # Ensure order
        
        for mod_name in self.modality_names:

            if len(self.modalities_config.get(mod_name)) > 1:
                self.channel_poolings[mod_name] = ChannelPooling(
                    pooling_type=self.pooling_type,
                    att_size=att_size
                )
        
        self.ecg_aggregator = nn.Linear(self.factor_resolution_ecg, 1)

        # Initialize CoupledMambaModel
        
        self.coupled_mamba = EfficientCoupledMambaModel(
            num_modalities=len(self.modality_names),
            **config.get('mamba_config', {})
        )

        self.proj_mods = nn.ModuleDict({
            mod_name.lower(): nn.LazyLinear(self.coupled_mamba.d_model)
            for mod_name in self.modality_names
        })

        self.dropout = nn.Dropout(p=config.get('dropout', 0))

        if self.use_demographic_features:
            
            # Use a small MLP for demographic features
            # Input dim will be inferred or passed. 
            # We use LazyLinear for the first layer to adapt to input dimension automatically
            self.demo_mlp = nn.Sequential(
                nn.LazyLinear(self.coupled_mamba.d_model // 2),
                nn.LayerNorm(self.coupled_mamba.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.coupled_mamba.d_model // 2, self.coupled_mamba.d_model // 4),
                nn.LayerNorm(self.coupled_mamba.d_model // 4),
                nn.ReLU()
            )
            # Input to head is d_model * num_modalities + d_model // 4
            self.head = nn.Linear(self.coupled_mamba.d_model*len(self.modality_names) + self.coupled_mamba.d_model//4, number_lables)
        else:
            self.head = nn.Linear(self.coupled_mamba.d_model*len(self.modality_names), number_lables) # instead of linear we could use nn.Parameter modules obtaining weighted sum
        
    
    def forward(self, inputs: dict, masks: dict, demo_features: Optional[list] = None):
        # inputs: dict {modality_name: (B, C, L_input_res_mod, D) 
        # masks: dict {modality_name: (B, L_input_res_mod) or None}
        # demo_features: list of (B, K) or None
        
        mamba_inputs = [] # Mamba elaborates list of inputs as different modalities
        for mod_name in self.modality_names:
            key = mod_name
            if key not in inputs:
                raise ValueError(f"Modality {key} not found in inputs")
                
            x = inputs[key]
            # embeddings (B, C, L_input_res_mod, D)
            # Apply Channel Pooling -> (B, L_input_res_mod, D)
            if x.shape[1] == 1:
                x_pooled = x.squeeze(1)
            elif key in self.channel_poolings:
                x_pooled = self.channel_poolings[key](x, masks[key])
            else:
                raise ValueError(f"Modality {key} not found in channel_poolings and more than 1 channel present")
        
            if self.resolution_input_embedding_s.get(key)==self.resolution_output_embedding_s:
                x = self.proj_mods[key](x_pooled) # (B, L_input_res_mod, D) -> (B, L_out_res, d_model)
            else:
                factor_resolution = int(self.resolution_output_embedding_s/self.resolution_input_embedding_s[key])
                if key == 'ecg':
                    x = self.proj_mods[key](x_pooled) # (B, L_input_res_mod, D) -> (B, L_input_res_mod, d_model)
                    # GAP over factor_resolution consecutive elements -> (B, L_input_res_mod/factor_resolution = L_output_res, d_model)
                    # x = rearrange(x, 'B (L f) D -> B L f D', f=factor_resolution).mean(dim=2)
                    # Instead of using GAP, aggregate with learnable linear layer
                    x = rearrange(x, 'B (L f) D -> B L D f', f=factor_resolution) # (B, L_input_res_mod/factor_resolution = L_output_res, D, factor_resolution)
                    x = self.ecg_aggregator(x).squeeze(-1) # (B, L_output_res, d_model)
                else:
                    x = rearrange(x_pooled, 'B (L f) D -> B L (D f)', f=factor_resolution) # (B, L_output_res, D)
                    x = self.proj_mods[key](x) # (B, L_output_res, D) -> (B, L_output_res, d_model)
            
            mamba_inputs.append(x)
        
        # Verify alignment
        # Check that all mamba_inputs have the same sequence length (dim 1)
        lengths = [t.shape[1] for t in mamba_inputs]
        if len(set(lengths)) > 1:
            raise ValueError(f"Mismatching sequence lengths in mamba inputs: {lengths}. "
                             f"All modalities must align to the same output resolution. "
                             f"Check resolution settings and data preprocessing.")

        # since the output resolution is the same for all the modalities, now I have just a single mask referred to output_resolution
        matches_out_resolution = [self.resolution_input_embedding_s.get(key)==self.resolution_output_embedding_s for key in self.modality_names]
        if any(matches_out_resolution):
            mamba_mask = masks[self.modality_names[np.where(matches_out_resolution)[0][0]]]
        else:
            input_lengths = torch.sum(masks[self.modality_names[0]], dim=1)
            factor_resolution = int(self.resolution_output_embedding_s/self.resolution_input_embedding_s[self.modality_names[0]])
            output_lengths = input_lengths//factor_resolution
            mamba_mask = torch.zeros((len(output_lengths), max(output_lengths)), dtype=torch.long, device=x.device)
            for i, length in enumerate(output_lengths):
                mamba_mask[i, :length] = 1
            
        # Pass to CoupledMambaModel
        # It expects list of inputs and list of masks
        out = self.coupled_mamba(mamba_inputs, mamba_mask)

        if self.use_demographic_features and demo_features is not None:
            # Process demographic features
            demographic_embedding = self.demo_mlp(demo_features)
            out = torch.cat([out, demographic_embedding], dim=1)

        out = self.dropout(out)

        # out = torch.relu(out) # relu used in coupled mamba paper github

        linear_projection = self.head(out)
        
        return linear_projection

class LitMultimodalModel(BaseSystem):
    def __init__(self, model_config, number_labels, weight_decay=1e-2, create_model=True, **kwargs):
        """
        Args:
            model_config (dict): Configuration for the MultimodalModel.
            number_labels (int): Number of output labels.
            create_model (bool): If True, instantiates MultimodalModel. Set False for subclasses that define their own model.
            **kwargs: Additional arguments for BaseSystem (metrics, log_filter, etc.).
        """
        # Initialize BaseSystem with required args
        super().__init__(**kwargs)
        
        # Save Hyperparameters (excluding model components which might not be picklable or needed)
        self.save_hyperparameters(ignore=[*kwargs]) 

        if create_model:
            self.model = MultimodalModel(model_config, number_labels)
        self.use_optimal_threshold = False
        self.weight_decay = weight_decay
        
    def configure_optimizers(self):
        # Selective weight decay: apply only to params NOT in coupled_mamba
        # Group parameters
        decay = []
        no_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if parameter belongs to coupled_mamba
            if "coupled_mamba" in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        # Create optimizer with groups
        optim_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        
        logger.info(f"Selective Weight Decay: {len(decay)} params with decay={self.weight_decay}, {len(no_decay)} params with decay=0.0")
        
        self.optimizer = optim.AdamW(optim_groups, lr=self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, total_steps = self.trainer.estimated_stepping_batches, pct_start = 0.1, anneal_strategy = 'cos')
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
    
    def setup(self, stage=None):
        """
        Initialize LazyLayers by running a dummy forward pass with a sample from the dataloader.
        This ensures parameters are initialized before the optimizer is created.
        """
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            try:
                # Retrieve the train dataloader to get one batch
                # We use the train dataloader as it should be available
                dl = self.trainer.datamodule.train_dataloader()
                if isinstance(dl, torch.utils.data.DataLoader):
                    # Helper to get one batch without disrupting main iterator if possible, 
                    # but creating a new iterator is safe here.
                    batch = next(iter(dl))
                    inputs = batch[0]
                    masks = batch[1]
                    
                    # Ensure device alignment
                    # Move batch to model device
                    device = self.device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    masks = {k: (v.to(device) if v is not None else None) for k, v in masks.items()}
                    
                    demo_features = None
                    if len(batch) >= 4 and batch[3] is not None:
                        demo_features = batch[3].to(device)

                    # Run forward in no_grad mode
                    with torch.no_grad():
                        _ = self(inputs, masks, demo_features)
                    
                    logger.info("LazyLinear layers initialized via dummy forward pass.")

                    logger.info("Applying explicit weight initialization...")
                    self.model.apply(init_weights)

                    if isinstance(self.logger, WandbLogger):
                        self.logger.watch(self, log='all', log_freq=50)

            except Exception as e:
                logger.info(f"Warning: Failed to initialize LazyLinear layers in setup(): {e}")    
    
    def forward(self, inputs, masks, demo_features=None):
        return self.model(inputs, masks, demo_features)
    
    @property
    def step_args(self):
        # Return the keys that _step will generate
        return ['logits', 'labels', 'hazards', 'event_time', 'is_event',]

    def _step(self, batch):
        # Unpack batch
        # Dataset returns: inputs, masks, labels, demo_features (optional), ids
        if len(batch) == 5:
            inputs, masks, labels, demo_features, _ = batch
        else:
            inputs, masks, labels, _ = batch
            demo_features = None

        if demo_features is not None and demo_features.numel() == 0:
            demo_features = None

        logits = self(inputs, masks, demo_features)
        
        step_output = {'logits': logits, 'labels': labels}

        # Check for survival task based on labels shape
        # Survival labels are expected to be [B, 1, 2] (binary survival) or [B, N_labels, 2] if multilabel
        if labels.dim() == 3:
            is_event = labels[:, :, 0]
            event_time = labels[:, :, 1]
            step_output['hazards'] = logits
            step_output['event_time'] = event_time
            step_output['is_event'] = is_event

        return step_output
    
    def update_threshold(self, threshold):
        self.use_optimal_threshold = True
        self.best_threshold = threshold
        
        # Clone test metrics with new prefix to avoid overwriting default test metrics
        if self.test_metrics:
            # We want to change the prefix from "test_" to "test_optimal_"
            # Current prefix is likely "test_" determined in BaseSystem
            self.test_metrics = self.test_metrics.clone(prefix='test_optimal_')
            
            for name, metric in self.test_metrics.items():
                if hasattr(metric, 'metric') and hasattr(metric.metric, 'threshold'):
                     # If it's our MetricWrapper
                     metric.metric.threshold = self.best_threshold
                elif hasattr(metric, 'threshold'):
                     # Direct metric
                     metric.threshold = self.best_threshold
    
    def change_dataset_test(self, dataset_name:str):
        self.test_metrics = self.test_metrics.clone(prefix=f'test_{dataset_name}')

class DemoBaselineModel(nn.Module):
    def __init__(self, n_demo_features, model_config, number_labels):
        super(DemoBaselineModel, self).__init__()
        mamba_config = model_config.get('mamba_config')
        d_model = mamba_config.get('d_model')
        self.demo_mlp = nn.Sequential(
                nn.Linear(n_demo_features, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU()
            )
        self.dropout = nn.Dropout(p=model_config.get('dropout', 0))
        self.disease_heads = nn.Linear(d_model // 4, number_labels)

    def forward(self, inputs, masks, demo_features):
        demo_embed = self.demo_mlp(demo_features)
        demo_embed = self.dropout(demo_embed)
        logits = self.disease_heads(demo_embed)
        return logits

class LitDemoBaselineModel(LitMultimodalModel):
    def __init__(self, n_demo_features, model_config, number_labels, **kwargs):
        super().__init__(model_config, number_labels, create_model=False, **kwargs)
        self.model = DemoBaselineModel(n_demo_features, model_config, number_labels)