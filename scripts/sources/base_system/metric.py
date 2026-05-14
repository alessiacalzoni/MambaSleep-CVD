from scripts.sources.base_system.base_system import BaseMetric
from typing import List
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
import torch
import logging

logger = logging.getLogger(__name__)

class ClassificationMetric(BaseMetric):
    def __init__(
        self,
        metric : object,
        input_args : List[str],
        batch_args : List[str] = [],
        int_args : List[str] = [],
        out_labels : List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.inp_args = input_args
        self.b_args = batch_args
        self.int_args = int_args
        self.metric = metric
        self.out_labels = out_labels

    @property
    def input_args(self):
        return self.inp_args

    @property
    def batch_args(self):
        return self.b_args

    def update(self, args):
        m_args = {k: v for k, v in args.items() if k in self.b_args}
        m_args.update({k: v for k, v in args.items() if k in self.inp_args})

        # Cast args to int if specified
        for k in self.int_args:
            m_args[k] = m_args[k].int()

        self.metric.update(*m_args.values())

    def compute(self):
        out = self.metric.compute()

        if not self.out_labels:
            return out

        if len(self.out_labels) != len(out):
            logger.info('Problems')

        res = {}
        for label, value in zip(self.out_labels, out):
            res[label] = value.detach().cpu()

        return res

class SurvivalMetric(BaseMetric):
    def __init__(
        self,
        input_args : List[str] = ['hazards', 'event_time', 'is_event'],
        batch_args : List[str] = [],
        average: str = 'macro',
        threshold: List[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.inp_args = input_args
        self.b_args = batch_args
        self.average = average.lower() if average else 'none'
        self.threshold = threshold
        
        # We accumulate all predictions to compute C-Index at the end of epoch 
        # because C-Index is not decomposable (cannot simple average batch c-indices for exact global c-index).
        # However, for training monitoring, averaging batch C-indices is common. 
        # TorchSurv ConcordanceIndex works on batches.
        # To strictly follow the "log in wandb both the macro and the micro value", 
        # we can compute C-Index per batch and average, or accumulate.
        # Let's accumulate for correctness if memory allows, or just average.
        # Given "BaseMetric" structure usually updates and computes... 
        # Let's use torchmetrics style state accumulation.
        
        self.add_state("hazards", default=[], dist_reduce_fx="cat")
        self.add_state("event_time", default=[], dist_reduce_fx="cat")
        self.add_state("is_event", default=[], dist_reduce_fx="cat")

    @property
    def input_args(self):
        return self.inp_args

    @property
    def batch_args(self):
        return self.b_args

    def update(self, args):
        self.hazards.append(args['hazards'])
        self.event_time.append(args['event_time'])
        self.is_event.append(args['is_event'])

    def compute(self):
        if not self.hazards:
            return torch.tensor(0.5)
            
        hazards = torch.cat(self.hazards, dim=0) # (N, num_labels)
        event_time = torch.cat(self.event_time, dim=0) # (N, num_labels)
        is_event_raw = torch.cat(self.is_event, dim=0) # (N, num_labels), keep int to detect -1
        
        if self.threshold:
            metric = Auc()
            if self.average == 'micro':
                try:
                    h_flat = hazards.view(-1)
                    t_flat = event_time.view(-1)
                    e_flat = is_event_raw.view(-1)
                    valid = e_flat >= 0
                    return metric(estimate = h_flat[valid], time = t_flat[valid], event = e_flat[valid].bool(), new_time = self.threshold)
                except ValueError:
                    return torch.tensor(0.5)
            elif self.average == 'macro' or self.average not in ['micro', 'macro']:
                scores = []
                num_labels = hazards.shape[1]
                for i in range(num_labels):
                    try:
                        valid = is_event_raw[:, i] >= 0
                        scores.append(metric(estimate = hazards[valid, i], time = event_time[valid, i], event = is_event_raw[valid, i].bool(), new_time = self.threshold))
                    except ValueError:
                        # Fallback for labels with no events in validation
                        scores.append(torch.tensor(0.5))
                if self.average == 'macro':
                    return torch.stack(scores).mean()
                else: # 'none' or 'per_label'
                    return torch.stack(scores)
        else:
            if self.average == 'micro':
                # Flatten everything
                h_flat = hazards.view(-1)
                t_flat = event_time.view(-1)
                e_flat = is_event_raw.view(-1)
                valid = e_flat >= 0
                return self._compute_cindex(h = h_flat[valid], t = t_flat[valid], e = e_flat[valid])
                
            elif self.average == 'macro':
                # Compute per label and mean
                scores = []
                num_labels = hazards.shape[1]
                for i in range(num_labels):
                    valid = is_event_raw[:, i] >= 0
                    scores.append(self._compute_cindex(h = hazards[valid, i], t = event_time[valid, i], e = is_event_raw[valid, i]))
                return torch.stack(scores).mean()
                
            else: # 'none' or 'per_label'
                # Return list/dict of scores
                scores = []
                num_labels = hazards.shape[1]
                for i in range(num_labels):
                    valid = is_event_raw[:, i] >= 0
                    scores.append(self._compute_cindex(h = hazards[valid, i], t = event_time[valid, i], e = is_event_raw[valid, i]))
                return torch.stack(scores)

    def _compute_cindex(self, h, t, e):
        # Ensure e is strictly binary 0/1
        e_binary = (e > 0).bool()
        
        if e_binary.sum() == 0:
             return torch.tensor(0.5, device=h.device)

        if ConcordanceIndex is None:
             return self._manual_cindex(h, t, e_binary)
             
        try:
             metric = ConcordanceIndex()
             # Pass e as float or int, sometimes libraries prefer that over bool
             return metric(h, e_binary, t)
        except Exception as err:
             logger.warning(f"TorchSurv calculation failed: {err}. Falling back to manual.")
             return self._manual_cindex(h, t, e_binary)

    def _manual_cindex(self, hazards, event_times, is_event):
        # ... (Previous manual implementation) ...
        # Ensure 1D
        hazards = hazards.view(-1)
        event_times = event_times.view(-1)
        is_event = is_event.view(-1)
        
        t_i = event_times.unsqueeze(1)
        t_j = event_times.unsqueeze(0)
        e_i = is_event.unsqueeze(1)
        
        valid_pairs = (t_i < t_j) & (e_i == 1)
        if valid_pairs.sum() == 0:
            return torch.tensor(0.5, device=hazards.device)
            
        h_i = hazards.unsqueeze(1)
        h_j = hazards.unsqueeze(0)
        
        concordant = (h_i > h_j) & valid_pairs
        ties = (h_i == h_j) & valid_pairs
        
        return (concordant.float().sum() + 0.5 * ties.float().sum()) / valid_pairs.float().sum()