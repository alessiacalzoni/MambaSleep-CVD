import torch
import torch.nn as nn
from scripts.sources.base_system.base_system import BaseLoss

class ClassificationLoss(BaseLoss):
    """
    Classification Loss module compatible with BaseSystem.
    Handles both binary and multilabel classification.
    """
    def __init__(self, gamma=None, alpha=None, pos_weight=None, reduction='mean'):
        """
        Args:
            gamma (int): The gamma value for the focal loss.
            alpha (int): The alpha value for the focal loss.
            pos_weight (tensor): The weight given to the positive class
                to deal with the class imbalance even more.
            reduction (str): Reduction method ('mean', 'sum', 'sum_mean').
        """
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        # We use reduction='none' so BaseLoss can handle reduction logic if needed, 
        # though BaseLoss.apply_reduction expects us to return per-sample loss mostly.
        # Therefore, I can get information also about the specific loss values (not aggregated)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    @property
    def input_args(self):
        return ['logits', 'labels']

    def forward(self, step_dict):
        logs = {}
        logits = step_dict['logits']
        labels = step_dict['labels']
             
        # If strictly binary (shape [B]), unsqueeze to [B, 1]
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        if (self.gamma is not None) & (self.alpha is not None):
            # FOCAL LOSS
            # ----------------
            # This is the vector of losses (per sample) because reduction='none'
            BCE_loss = self.criterion(logits, labels.float())
            # Clamp BCE loss to avoid infinity (which causes JSON errors in logging)
            BCE_loss = torch.clamp(BCE_loss, min=0, max=100)
            # prevents nans when probability 0
            pt = torch.exp(-BCE_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
            # Clamp focal loss just in case
            loss = torch.clamp(focal_loss, min=0, max=100)
        else:
            loss = self.criterion(logits, labels.float())

        loss_reduced = self.apply_reduction(loss)
        
        # # Populate logs with stats of the unreduced loss or the vector itself (if logger supports it)
        # # Note: Standard Lightning loggers might error on vectors, so we log stats here.
        # logs = {
        #     'loss_max': loss_unreduced.max(),
        #     'loss_std': loss_unreduced.std(),
        #     # 'loss_unreduced': loss_unreduced # Uncomment if using a logger that supports vectors/histograms
        # }
        
        return loss_reduced, logs

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

def stratified_cox_ph_loss(hazards, event_times, is_event, stratum_ids, min_events=3, valid_mask=None, pos_weight=None):
    """CoxPH loss computed within each age stratum. Forces the model to rank
    patients relative to peers of the same age, not across age groups.

    Args:
        hazards:     (B, L) predicted log-hazards (signal + offset)
        event_times: (B, L) time to event or censoring
        is_event:    (B, L) binary event indicator
        stratum_ids: (B,)   integer age-bin id per sample
        min_events:  skip strata with fewer events than this

    Returns:
        Scalar loss averaged over valid strata and labels.
    """
    unique_strata = stratum_ids.unique()
    stratum_losses = []

    for s in unique_strata:
        mask = stratum_ids == s
        h_s  = hazards[mask]        # (Ns, L)
        t_s  = event_times[mask]    # (Ns, L)
        e_s  = is_event[mask]       # (Ns, L)
        v_s  = valid_mask[mask] if valid_mask is not None else None

        # Need enough events per label on average to form a valid loss
        if e_s.clamp(min=0).sum() < min_events:
            continue

        # Compute standard CoxPH loss within this stratum
        stratum_losses.append(cox_ph_loss(h_s, t_s, e_s, valid_mask=v_s, pos_weight=pos_weight))

    if len(stratum_losses) == 0:
        return hazards.sum() * 0.0  # zero loss, keeps graph alive

    return torch.stack(stratum_losses).mean()


class StratifiedSurvivalLoss(BaseLoss):
    """Survival loss that computes CoxPH within age strata."""

    def __init__(self, pos_weight=None, min_events=10, reduction='mean'):
        super().__init__(reduction=reduction)
        self.min_events = min_events
        if pos_weight is not None:
             self.register_buffer('pos_weight', pos_weight)
        else:
             self.pos_weight = None

    @property
    def input_args(self):
        return ['hazards', 'event_time', 'is_event', 'stratum_ids']

    def forward(self, step_dict):
        valid_mask = step_dict['is_event'] >= 0
        is_event_clean = step_dict['is_event'].clamp(min=0)
        
        loss = stratified_cox_ph_loss(
            step_dict['hazards'],
            step_dict['event_time'],
            is_event_clean,
            step_dict['stratum_ids'],
            min_events=self.min_events,
            valid_mask=valid_mask,
            pos_weight=self.pos_weight,
        )
        return loss, {}


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
