import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from einops import rearrange

class PatchEmbedECG(nn.Module):
    def __init__(self, in_channels, d_model, Fs, R_out):
        """
        CNN-based tokenizer for raw ECG signals using 3 Conv1d layers.
        
        Total downsampling stride (patch_size) = Fs * R_out.
        The stride is factored across 3 layers so that s1 * s2 * s3 = patch_size.
        
        Kernel sizes:
          - Layer 1: kernel_size = 7
          - Layer 2: kernel_size = 5
          - Layer 3: kernel_size = s3 (computed dynamically for alignment)
        
        If the factorization is not exact, an adaptive average pool is used
        to enforce the correct output length.
        """
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = int(Fs * R_out)
        
        # Factor total stride across 3 layers
        s1 = int(math.ceil(self.patch_size ** (1/3)))
        s2 = int(math.ceil((self.patch_size / s1) ** (1/2)))
        s3 = self.patch_size // (s1 * s2)
        # Ensure s3 >= 1
        if s3 < 1:
            s3 = 1
        
        self.strides = (s1, s2, s3)
        self.actual_stride = s1 * s2 * s3
        
        if self.actual_stride != self.patch_size:
            logger.warning(
                f"ECG Tokenizer: Factored stride {self.actual_stride} "
                f"(={s1}*{s2}*{s3}) != target patch_size {self.patch_size}. "
                f"Adaptive pooling will be used to correct the output length."
            )
        
        # Kernel size for the 3rd layer: match the stride for proper coverage
        k3 = s3 if s3 % 2 == 1 else s3 + 1  # ensure odd kernel for symmetric padding

        # Layer 1: kernel=7, stride=s1
        self.conv1 = nn.Conv1d(in_channels, d_model // 4, kernel_size=7, stride=s1, padding=3)
        self.bn1 = nn.BatchNorm1d(d_model // 4)

        # Layer 2: kernel=5, stride=s2
        self.conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=5, stride=s2, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model // 2)

        # Layer 3: kernel=k3, stride=s3 — computed for alignment
        self.conv3 = nn.Conv1d(d_model // 2, d_model, kernel_size=k3, stride=s3, padding=k3 // 2)
        
        self.final_proj = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        
        logger.info(
            f"ECG Tokenizer: patch_size={self.patch_size}, "
            f"strides=({s1},{s2},{s3}), kernels=(7,5,{k3}), "
            f"actual_stride={self.actual_stride}"
        )

    def forward(self, x):
        """
        x: [B, C, L]
        returns: [B, num_patches, d_model]
        """
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # [B, d_model, L']
        
        # If factorization was not exact, enforce correct output length
        if self.actual_stride != self.patch_size:
            target_len = x.shape[-1] * self.actual_stride // self.patch_size
            if target_len < 1:
                target_len = 1
            x = F.adaptive_avg_pool1d(x, target_len)

        tokens = x.transpose(1, 2)  # [B, L', d_model]
        tokens = self.final_proj(tokens)
        tokens = self.layernorm(tokens)
        return tokens

class PatchEmbedEEG(nn.Module):
    def __init__(self, in_channels, d_model, Fs, R_out):
        """
        Adapted from EEGMamba tokenizer.
        Total downsampling = patch_size = Fs * R_out
        """
        super().__init__()
        self.d_model = d_model
        self.patch_size = int(Fs * R_out)
        self.in_channels = in_channels
        
        # Position encoding applied per channel then summed? EEGMamba applies it to (B, d_model, C, L)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3),
                      groups=d_model, bias=False),
        )

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model//2, kernel_size=(1, self.patch_size), stride=(1, self.patch_size), bias=False),
            nn.GroupNorm(4, d_model//2),
            nn.GELU(),
        )
        
        # FFT branch
        self.spectral_proj = nn.Sequential(
            nn.Linear(self.patch_size // 2 + 1, d_model//2, bias=False),
            nn.Dropout(0.1),
        )
        
        self.final_proj = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, C, L)
        Returns: (B, L_out, d_model)
        """
        bz, ch_num, seq_len = x.shape
        
        # Ensure L is perfectly divisible by patch_size for simplicity, or truncate
        num_patches = seq_len // self.patch_size
        x = x[:, :, :num_patches * self.patch_size].contiguous()
        
        # Reshape to (B, C, num_patches, patch_size)
        x_reshaped = x.view(bz, ch_num, num_patches, self.patch_size)
        
        # --- Time Branch ---
        # proj_in expects (B, 1, C*num_patches, patch_size)
        time_x = x_reshaped.view(bz, 1, ch_num * num_patches, self.patch_size)
        time_emb = self.proj_in(time_x) # (B, d_model//2, C*num_patches, 1)
        time_emb = time_emb.view(bz, self.d_model//2, ch_num, num_patches).permute(0, 2, 3, 1) # (B, C, num_patches, d_model//2)

        # --- Freq Branch ---
        spectral = torch.fft.rfft(x_reshaped, dim=-1, norm='forward') # (B, C, num_patches, patch_size//2 + 1)
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral) # (B, C, num_patches, d_model//2)
        
        # Combine
        patch_emb = torch.cat([time_emb, spectral_emb], dim=-1) # (B, C, num_patches, d_model)
        
        # Positional Encoding 
        # Needs (B, d_model, C, num_patches)
        patch_emb = patch_emb.permute(0, 3, 1, 2)
        patch_emb = patch_emb + self.positional_encoding(patch_emb)
        
        # Channel pooling: Mean across channels
        patch_emb = patch_emb.mean(dim=2) # (B, d_model, num_patches)
        
        tokens = patch_emb.transpose(1, 2) # (B, num_patches, d_model)
        tokens = self.final_proj(tokens)
        tokens = self.layernorm(tokens)
        return tokens

class PatchEmbedEOG(nn.Module):
    def __init__(self, in_channels, d_model, Fs, R_out):
        """
        EOGNET-inspired dual-scale CNN tokenizer for raw EOG signals.
        (Fan et al., 2021 — Frontiers in Neuroscience)
        
        Total downsampling (patch_size) = Fs * R_out.
        
        Two parallel branches extract features at different temporal scales:
          - Branch 1 (high-freq): kernel ≈ 0.5 * Fs, small stride
            → captures rapid eye movements, blinks, saccades
          - Branch 2 (low-freq):  kernel ≈ 5 * Fs,   large stride
            → captures slow oscillations and EEG-contaminated sleep waves
        
        Features from both branches are concatenated and projected to d_model.
        Adaptive pooling enforces the target output length (L / patch_size).
        """
        super().__init__()
        self.d_model = d_model
        Fs = int(Fs)
        self.patch_size = int(Fs * R_out)
        
        # =====================================================================
        # Branch 1: High-frequency (small kernel, small stride)
        # kernel ≈ Fs/2 → ~10ms temporal resolution at 50Hz 
        # stride ≈ Fs/8 → moderate downsampling per layer
        # =====================================================================
        k1_small = max(3, Fs // 2)
        if k1_small % 2 == 0:
            k1_small += 1  # ensure odd
        s1_small = max(1, Fs // 8)
        
        # Two conv layers in the small-filter branch
        self.small_conv1 = nn.Conv1d(in_channels, d_model // 4, kernel_size=k1_small, stride=s1_small, padding=k1_small // 2)
        self.small_bn1 = nn.BatchNorm1d(d_model // 4)
        
        # Second layer: smaller kernel for refinement
        k2_small = max(3, k1_small // 2)
        if k2_small % 2 == 0:
            k2_small += 1
        s2_small = max(1, s1_small)  # same stride as first layer
        
        self.small_conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=k2_small, stride=s2_small, padding=k2_small // 2)
        self.small_bn2 = nn.BatchNorm1d(d_model // 2)
        
        self.small_stride = s1_small * s2_small
        
        # =====================================================================
        # Branch 2: Low-frequency (large kernel, large stride)
        # kernel ≈ 5*Fs → captures patterns spanning ~5 seconds
        # stride ≈ Fs → aggressive downsampling (1 step per second)
        # =====================================================================
        k1_large = max(3, Fs * 5)
        if k1_large % 2 == 0:
            k1_large += 1  # ensure odd
        s1_large = max(1, Fs)
        
        self.large_conv1 = nn.Conv1d(in_channels, d_model // 4, kernel_size=k1_large, stride=s1_large, padding=k1_large // 2)
        self.large_bn1 = nn.BatchNorm1d(d_model // 4)
        
        # Second layer: smaller kernel for refinement
        k2_large = 7  # fixed small kernel
        s2_large = max(1, self.patch_size // (s1_large * s1_small))  # absorb remaining stride
        if s2_large < 1:
            s2_large = 1
        
        self.large_conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=k2_large, stride=s2_large, padding=k2_large // 2)
        self.large_bn2 = nn.BatchNorm1d(d_model // 2)
        
        self.large_stride = s1_large * s2_large
        
        # =====================================================================
        # Fusion: concatenate both branches → project to d_model
        # =====================================================================
        self.fusion_proj = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        
        logger.info(
            f"EOG Tokenizer (EOGNET-inspired): patch_size={self.patch_size}, Fs={Fs}\n"
            f"  Branch 1 (high-freq): kernels=({k1_small},{k2_small}), strides=({s1_small},{s2_small}), total_stride={self.small_stride}\n"
            f"  Branch 2 (low-freq):  kernels=({k1_large},{k2_large}), strides=({s1_large},{s2_large}), total_stride={self.large_stride}"
        )

    def forward(self, x):
        """
        x: (B, C, L)
        returns: (B, num_patches, d_model)
        """
        target_len = max(1, x.shape[-1] // self.patch_size)
        
        # --- Branch 1: high-frequency features ---
        h_small = F.gelu(self.small_bn1(self.small_conv1(x)))
        h_small = F.gelu(self.small_bn2(self.small_conv2(h_small)))  # (B, d_model//2, L1)
        # Align to target length
        h_small = F.adaptive_avg_pool1d(h_small, target_len)  # (B, d_model//2, target_len)
        
        # --- Branch 2: low-frequency features ---
        h_large = F.gelu(self.large_bn1(self.large_conv1(x)))
        h_large = F.gelu(self.large_bn2(self.large_conv2(h_large)))  # (B, d_model//2, L2)
        # Align to target length
        h_large = F.adaptive_avg_pool1d(h_large, target_len)  # (B, d_model//2, target_len)
        
        # --- Concatenate and project ---
        h = torch.cat([h_small, h_large], dim=1)  # (B, d_model, target_len)
        tokens = h.transpose(1, 2)  # (B, target_len, d_model)
        tokens = self.fusion_proj(tokens)
        tokens = self.layernorm(tokens)
        return tokens

from scripts.sources.base_system.base_system import BaseSystem
from scripts.sources.models.efficient_coupled_mamba import EfficientCoupledMambaModel
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger

class RawMultimodalModel(nn.Module):
    def __init__(self, config, number_lables: int):
        super().__init__()
        
        self.modalities_config = config['modalities']
        self.use_demographic_features = config.get('use_demographic_features', False)
        self.resolution_output_embedding_s = config.get('resolution_output_embedding_s')
        
        # Sampling rates: prefer direct 'sampling_rate_hz' (e.g., {"ecg": 250})
        # Fallback to legacy 'resolution_input_embedding_s' with Fs = 1 / T_s
        self.sampling_rate_hz = config.get('sampling_rate_hz', None)
        if self.sampling_rate_hz is None:
            res_input = config.get('resolution_input_embedding_s', {})
            self.sampling_rate_hz = {mod: int(1 / res) for mod, res in res_input.items()}
            logger.warning("Using legacy 'resolution_input_embedding_s' → converted to sampling_rate_hz. "
                           "Consider switching to 'sampling_rate_hz' in your config.")
        
        self.modality_names = list(self.modalities_config.keys())
        
        mamba_config = config.get('mamba_config', {})
        d_model = mamba_config.get('d_model', 128)
        
        self.patch_embeds = nn.ModuleDict()
        for mod_name in self.modality_names:
            in_channels = len(self.modalities_config[mod_name])
            Fs = int(self.sampling_rate_hz[mod_name])
            R_out = self.resolution_output_embedding_s
            
            if mod_name == 'ecg':
                self.patch_embeds[mod_name] = PatchEmbedECG(in_channels, d_model, Fs, R_out)
            elif mod_name == 'eeg':
                self.patch_embeds[mod_name] = PatchEmbedEEG(in_channels, d_model, Fs, R_out)
            elif mod_name == 'eog':
                self.patch_embeds[mod_name] = PatchEmbedEOG(in_channels, d_model, Fs, R_out)
            else:
                raise ValueError(f"Unknown modality {mod_name}")

        self.coupled_mamba = EfficientCoupledMambaModel(
            num_modalities=len(self.modality_names),
            **mamba_config
        )

        self.dropout = nn.Dropout(p=config.get('dropout', 0))

        if self.use_demographic_features:
            self.demo_mlp = nn.Sequential(
                nn.LazyLinear(d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU()
            )
            self.head = nn.Linear(d_model*len(self.modality_names) + d_model//4, number_lables)
        else:
            self.head = nn.Linear(d_model*len(self.modality_names), number_lables)
        
        # When True, backbone forward pass runs in torch.no_grad() to save GPU memory
        # Set to True during fine-tuning when backbone is frozen
        self.backbone_no_grad = False
    def forward(self, inputs: dict, masks: dict, demo_features=None):
        # When backbone is frozen, force no_grad to save memory.
        # Otherwise, use a null context so the CALLER's context controls gradients:
        # during training gradients are enabled by default; during Lightning's
        # validation torch.no_grad() is active. Using enable_grad() here would
        # override that, causing autograd graphs to accumulate without ever being
        # freed (no backward call in validation) → GPU OOM.
        backbone_ctx = torch.no_grad() if self.backbone_no_grad else contextlib.nullcontext()

        with backbone_ctx:
            mamba_inputs = []
            for mod_name in self.modality_names:
                x = inputs[mod_name] # (B, C, L)
                
                # Tokenize into (B, L_out, d_model)
                tokens = self.patch_embeds[mod_name](x)
                mamba_inputs.append(tokens)
            
            # Different tokenizers (EEG view-based, ECG CNN-based, EOG pooling-based)
            # may produce token sequences that differ by ±1 due to integer rounding.
            # Align all to the minimum length.
            L_out = min(t.shape[1] for t in mamba_inputs)
            mamba_inputs = [t[:, :L_out, :] for t in mamba_inputs]
            
            ref_mod = self.modality_names[0]
            
            # Subsample mask
            # original mask: (B, L_in). We want (B, L_out)
            # stride = L_in / L_out
            original_mask = masks[ref_mod] # (B, L_in)
            stride = original_mask.shape[1] // L_out
            if stride == 0: stride = 1
            
            # Taking every `stride` element
            mamba_mask = original_mask[:, ::stride][:, :L_out]
            
            # Pass to Mamba
            out = self.coupled_mamba(mamba_inputs, mamba_mask)

        # Head runs WITH gradients (it's trainable)
        if self.use_demographic_features and demo_features is not None:
            demographic_embedding = self.demo_mlp(demo_features)
            out = torch.cat([out, demographic_embedding], dim=1)

        out = self.dropout(out)
        linear_projection = self.head(out)
        
        return linear_projection

class LitRawMultimodalModel(BaseSystem):
    def __init__(self, model_config, number_labels, weight_decay=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=[*kwargs]) 
        self.model = RawMultimodalModel(model_config, number_labels)
        self.use_optimal_threshold = False
        self.weight_decay = weight_decay
        
    def configure_optimizers(self):
        decay = []
        no_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "coupled_mamba" in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        optim_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        
        self.optimizer = optim.AdamW(optim_groups, lr=self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, total_steps = self.trainer.estimated_stepping_batches, pct_start = 0.1, anneal_strategy = 'cos')
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler": self.scheduler, "interval": "step"}}
    
    def setup(self, stage=None):
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            try:
                dl = self.trainer.datamodule.train_dataloader()
                if isinstance(dl, torch.utils.data.DataLoader):
                    batch = next(iter(dl))
                    inputs = batch[0]
                    masks = batch[1]
                    
                    device = self.device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    masks = {k: (v.to(device) if v is not None else None) for k, v in masks.items()}
                    
                    demo_features = None
                    if len(batch) >= 4 and batch[3] is not None:
                        demo_features = batch[3].to(device)

                    with torch.no_grad():
                        _ = self(inputs, masks, demo_features)
                    
                    logger.info("LazyLinear layers initialized via dummy forward pass.")
            except Exception as e:
                logger.info(f"Warning: Failed to initialize LazyLinear layers in setup(): {e}")    
    
    def forward(self, inputs, masks, demo_features=None):
        return self.model(inputs, masks, demo_features)
    
    @property
    def step_args(self):
        return ['logits', 'labels', 'hazards', 'event_time', 'is_event', 'stratum_ids']

    def _step(self, batch):
        # Base dataset returns 5-tuple: (inputs, masks, labels, demo_features, ids)
        # Stratified dataset returns 6-tuple: (inputs, masks, labels, demo_features, stratum_ids, ids)
        if len(batch) == 5:
            inputs, masks, labels, demo_features, ids = batch
            stratum_ids = None
        else:
            inputs, masks, labels, demo_features, stratum_ids, ids = batch

        # demo_features may be None
        if demo_features is not None and demo_features.numel() == 0:
            demo_features = None

        logits = self(inputs, masks, demo_features)
        step_output = {'logits': logits, 'labels': labels}

        if labels.dim() == 3:
            is_event = labels[:, :, 0]
            event_time = labels[:, :, 1]
            step_output['hazards'] = logits
            step_output['event_time'] = event_time
            step_output['is_event'] = is_event
            if stratum_ids is not None:
                step_output['stratum_ids'] = stratum_ids

        return step_output
