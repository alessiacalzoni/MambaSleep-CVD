import torch
import torch.nn as nn
from scripts.sources.base_system.base_system import BaseLoss

def cox_ph_loss(hazards, event_times, is_event, valid_mask=None, pos_weight=None):
    B, L = hazards.shape
    if valid_mask is None:
        valid_mask = torch.ones(B, L, dtype=torch.bool, device=hazards.device)

    label_losses = []
    for k in range(L):
        v_k = valid_mask[:, k]                # (B,) bool
        h_k = hazards[v_k, k]                 # (n_valid,)
        t_k = event_times[v_k, k]             # (n_valid,)
        e_k = is_event[v_k, k]                # (n_valid,)

        if e_k.sum() < 1:
            continue

        t_k, idx = t_k.sort(descending=True)
        h_k = h_k[idx]
        e_k = e_k[idx]

        log_cum_h = torch.logcumsumexp(h_k.float(), dim=0)
        loss_k = -(h_k - log_cum_h) * e_k
        loss_k = loss_k.sum() / (e_k.sum() + 1e-9)

        if pos_weight is not None:
            loss_k = loss_k * pos_weight[k]
        label_losses.append(loss_k)

    if len(label_losses) == 0:
        return hazards.sum() * 0.0   # zero loss, keeps autograd graph alive

    return torch.stack(label_losses).mean()


class SurvivalLoss(BaseLoss):
    """
    Survival Loss module (CoxPH) compatible with BaseSystem.
    Expects logits (risk scores) and labels (containing both time and event).
    """
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__(reduction=reduction)
        # We don't initialize a PyTorch criterion because we implement CoxPH manually or use a library if available.
        # Implementing a basic Negative Log Partial Likelihood.
        if pos_weight is not None:
             self.register_buffer('pos_weight', pos_weight)
        else:
             self.pos_weight = None

    @property
    def input_args(self):
        return ['hazards', 'event_time', 'is_event']

    def forward(self, step_dict):
        hazards = step_dict['hazards']
        event_time = step_dict['event_time']
        is_event = step_dict['is_event']
        
        valid_mask = is_event >= 0
        is_event_clean = is_event.clamp(min=0)

        # from torchsurv.loss import cox
        # from torchsurv.metrics.cindex import ConcordanceIndex

        loss = cox_ph_loss(hazards, event_time, is_event_clean, valid_mask=valid_mask, pos_weight=self.pos_weight)
        
        return loss, {}
