import torch
from torch.utils.data import Sampler
import numpy as np

class SmartBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True, ensure_positive=False):
        """
        Args:
            data_source: Dataset or list containing lengths (must implement __len__ or be a list of lengths).
                         If it's the dataset, we assume it has a .lengths attribute or similar.
                         However, for flexibility, we'll assume we pass a list/dict of lengths separately if needed,
                         but here we design it to take the dataset and assume the dataset has a way to provide lengths.
            batch_size: Size of mini-batch.
            shuffle: Whether to shuffle the batches.
            ensure_positive: If True, ensures at least one positive sample in every batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # We assume data_source has a 'get_lengths' method or 'lengths' attribute
        # If not, we fall back to iterating (which is slow, so we expect pre-computed lengths)
        if hasattr(data_source, 'lengths'):
             self.lengths = data_source.lengths
        else:
             raise ValueError("Dataset must have 'lengths' attribute for SmartBatchSampler")
        
        self.labels = data_source.labels
        # Sort indices by length
        # We want to group by length, but also allow some randomness if needed.
        # But 'Smart Batching' usually implies strict length sorting for efficiency.
        # We sort indices, then chunk them into batches.
        # To add randomness (shuffle=True), we shuffle the *batches*, not the samples within batches.
        
        self.ensure_positive = ensure_positive
        
        # Sort indices by length
        self.sorted_indices = np.argsort(self.lengths)
        
    def __iter__(self):
        # Create batches from sorted indices
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i + self.batch_size]
            batches.append(batch)
            
        if self.shuffle:
            # Shuffle the order of batches, but not the contents yet (contents are length-sorted)
            np.random.shuffle(batches)
            
        final_batches = []
        
        if not self.ensure_positive:
            for batch in batches:
                yield batch
        else:
            # Pre-calculate positive indices in the entire dataset for efficient searching
            # Handle multilabel (any label > 0 is positive) or binary
            
            # Ensure labels are numpy array
            if hasattr(self.labels, "values"):
                 labels_np = self.labels.values
            elif isinstance(self.labels, torch.Tensor):
                 labels_np = self.labels.numpy()
            else:
                 labels_np = np.array(self.labels)

            if labels_np.ndim == 1:
                labels_np = labels_np[:, None] # Treat as (N, 1)

            n_samples, n_labels = labels_np.shape
            
            # Store indices of positives for each label
            pos_indices_per_label = []
            for k in range(n_labels):
                pos_indices_per_label.append(np.where(labels_np[:, k] > 0)[0])

            for batch in batches:
                batch = list(batch) # Convert to list for modification
                
                # Check coverage for each label
                current_batch_labels = labels_np[batch] # (batch_size, n_labels)
                
                # Check which labels are missing (sum of column <= 0)
                missing_labels = [k for k in range(n_labels) if current_batch_labels[:, k].sum() == 0]
                
                if missing_labels:
                    # For each missing label, we need to add a sample.
                    # We should try to replace a sample in the batch.
                    
                    # Compute batch mean length for finding replacements
                    batch_mean_len = np.mean([self.lengths[idx] for idx in batch])
                    
                    for missing_k in missing_labels:
                        # Check again if it's still missing (might have been covered by a previous replacement)
                        current_batch_labels = labels_np[batch]
                        if current_batch_labels[:, missing_k].sum() > 0:
                            continue

                        # Find a replacement for missing_k
                        candidates = pos_indices_per_label[missing_k]
                        if len(candidates) == 0:
                            # If no positives exist for this label in the whole dataset, we can't do anything
                            continue
                            
                        # Find candidate closest in length
                        cand_lengths = np.array(self.lengths)[candidates]
                        closest_arg = (np.abs(cand_lengths - batch_mean_len)).argmin()
                        best_cand_idx = candidates[closest_arg]
                        
                        # Find a victim to replace
                        # Strategy: replace a sample that is NOT critical for other labels.
                        # Calculate criticality:
                        # A sample is critical for label L if it is the ONLY positive for L in the batch.
                        
                        batch_label_sums = current_batch_labels.sum(axis=0)
                        
                        non_critical_victim_indices = []
                        for i, batch_idx in enumerate(batch):
                            # Check if this sample is critical for any label
                            is_critical = False
                            sample_labels = labels_np[batch_idx]
                            for label_idx in range(n_labels):
                                if sample_labels[label_idx] > 0 and batch_label_sums[label_idx] == 1:
                                    is_critical = True
                                    break
                            if not is_critical:
                                non_critical_victim_indices.append(i)
                        
                        if non_critical_victim_indices:
                            # Pick a random non-critical victim
                            victim_pos = np.random.choice(non_critical_victim_indices)
                        else:
                            # If all are critical (rare), just pick random
                            victim_pos = np.random.randint(0, len(batch))
                            
                        batch[victim_pos] = best_cand_idx
                
                final_batches.append(batch)
                
            for batch in final_batches:
                yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size
