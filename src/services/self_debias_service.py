"""
Service for Self-Debiasing framework implementation.

This service implements the self-debiasing approach from "Towards Debiasing NLU Models from 
Unknown Biases" which uses a shallow model to detect biases and reweight training examples.
"""

import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedModel
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, SequentialSampler, Subset

from config import TrainingConfig

logger = logging.getLogger(__name__)


class SelfDebiasService:
    """
    Service for implementing the self-debiasing framework.
    
    The approach involves:
    1. Training/finding a shallow model ($f_b$) on a small subset
    2. Using the shallow model to score all training examples
    3. Computing bias scores ($p_b^{(i,c)}$) for each example
    4. Using these scores to reweight the loss during main model training
    """
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.selfdebias_config = config.selfdebias
        self.device = device
        
        # Create cache directory for bias scores
        self.cache_dir = Path(config.outputdir) / "selfdebias_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_annealing_factor(self, current_step: int, total_steps: int, a: float = 0.8) -> float:
        """
        Compute the annealing factor alpha_t at the current training step.
        
        Formula: α_t = 1 - t * (1 - a) / T
        where:
        - t = current training step (clamped to [0, T])
        - T = total training steps 
        - a = minimum alpha value (default 0.8)
        
        Args:
            current_step: Current training step
            total_steps: Total number of training steps
            a: Minimum alpha value (default 0.8 from paper)
            
        Returns:
            The annealing factor alpha_t in [a, 1.0]
        """
        if total_steps <= 0:
            return 1.0
            
        # Clamp step to valid range
        current_step = max(0, min(current_step, total_steps))
        
        # Linear interpolation from 1.0 to a
        alpha_t = 1.0 - current_step * (1.0 - a) / total_steps
        
        # Safety clamp to ensure a <= alpha_t <= 1.0
        return float(max(a, min(1.0, alpha_t)))
    
    def anneal_probability_distribution(
        self, 
        prob_dist: torch.Tensor, 
        alpha: float,
        eps: float = 1e-12,
        dim: int = -1
    ) -> torch.Tensor:
        """
        Apply temperature scaling to a probability distribution using annealing.
        
        Formula: p_new(j) = p(j)^alpha / sum_k(p(k)^alpha)
        
        This flattens the distribution as alpha decreases, making it more uniform.
        
        Args:
            prob_dist: Probability distribution of shape [batch_size, num_labels]
            alpha: Temperature/annealing factor
            eps: Small constant for numerical stability
            dim: Dimension to normalize over (default: -1 for last dim)
            
        Returns:
            Annealed probability distribution that sums to 1.0
        """
        # Ensure valid probabilities
        prob_dist = prob_dist.clamp(min=eps)
        
        # Safety clamp alpha 
        alpha = max(0.0, min(100.0, float(alpha)))
        
        # Apply temperature scaling: p' = p^alpha
        annealed = torch.pow(prob_dist, alpha)
        
        # Normalize ensuring sum = 1.0
        normalizer = annealed.sum(dim=dim, keepdim=True).clamp(min=eps)
        annealed = annealed / normalizer
        
        return annealed
    

    def _resolve_base_dataset(self, ds):
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds

    def _resolve_orig_index(self, ds, i):
        # mappa l'indice i del dataset (potenzialmente Subset annidato) all'indice dell'originale
        orig = i
        d = ds
        while isinstance(d, Subset):
            orig = d.indices[orig]
            d = d.dataset
        return orig

    def get_full_probability_distributions(self, shallow_model, dataset, num_labels, device, batch_size, num_workers=0, collate_fn=None):
        # loader deterministico, NO shuffle
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        base_ds = self._resolve_base_dataset(dataset)
        N_base = len(base_ds)

        # qui metteremo le probs già allineate all'INDICE ORIGINALE del dataset base
        bias_probs_full = torch.zeros((N_base, num_labels), dtype=torch.float32)
        labels_all = torch.empty(N_base, dtype=torch.long)

        shallow_model.eval()
        with torch.no_grad():
            offset = 0
            for batch_idx, batch in tqdm(enumerate(loader), desc="Computing bias probabilities", total=len(loader)):
                # forward shallow
                inputs = {k: v.to(device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask', 'token_type_ids'] or torch.is_tensor(v)}
                logits = shallow_model(**{k: inputs[k] for k in inputs if k in ['input_ids','attention_mask','token_type_ids']}).logits
                probs = torch.softmax(logits, dim=-1).cpu()

                # ricava gli indici *locali* di questo batch nel dataset corrente
                bsz = probs.size(0)
                # qui calcoliamo gli indici esatti che il dataloader sta visitando: con SequentialSampler
                # sono semplicemente offset..offset+bsz-1
                idxs_local = torch.arange(offset, offset + bsz)
                offset += bsz

                # mappali agli indici ORIGINALI del dataset base
                orig_idx = torch.as_tensor([self._resolve_orig_index(dataset, int(i)) for i in idxs_local])

                # salva allineato
                bias_probs_full[orig_idx] = probs
                labels_all[orig_idx] = batch['labels'].cpu() if 'labels' in batch else batch[1].cpu()

        return bias_probs_full, labels_all
    
    
    
    def compute_reweighted_loss_with_full_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        bias_probs_full: torch.Tensor,
        current_step: int,
        total_steps: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Ensure inputs are on correct device
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        bias_probs_full = bias_probs_full.to(self.device)

        # 1. Compute annealing factor (alpha_t, or 'current_theta' in their code)
        alpha = self.compute_annealing_factor(
            current_step=current_step,
            total_steps=total_steps,
            a=self.selfdebias_config.annealing_min
        )
        
        with torch.no_grad():
            # 2. Extract the *original* probability for the correct label (p_b^(i,c))
            # This is equivalent to (one_hot_labels * teacher_probs).sum(1) in their code
            bias_prob_correct = bias_probs_full.gather(1, labels.unsqueeze(1)).squeeze(1)
            
            # 3. Compute the base weight: w = (1 - p_b^(i,c))
            base_weights = (1.0 - bias_prob_correct)
            
            # 4. Apply annealing exponent *directly to the weight*: w' = w^alpha
            # This is the key difference from your previous implementation.
            annealed_weights = base_weights ** alpha
            
            # 5. Detach for loss calculation
            weights = annealed_weights.detach()
        
        # Compute cross entropy loss per-sample
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply weights and normalize by the sum of weights (as in their code)
        weighted_loss = weights * ce_loss
        
        # Add clamp for numerical stability (prevents division by zero if all weights are 0)
        loss = weighted_loss.sum() / weights.sum().clamp(min=1e-8) 
        
        # Compute metadata
        with torch.no_grad():
            mean_weight = weights.mean().item()
            std_weight = weights.std().item()
            min_weight = weights.min().item()
            max_weight = weights.max().item()
            mean_bias_prob = bias_prob_correct.mean().item()
            std_bias_prob = bias_prob_correct.std().item()
            
        
        metadata = {
            # Training dynamics
            "alpha": alpha,
            "step": current_step,
            "total_steps": total_steps,
            
            # Weight statistics  
            "mean_weight": mean_weight,
            "std_weight": std_weight,
            "min_weight": min_weight,
            "max_weight": max_weight,
            
            # Bias probability statistics
            "mean_bias_prob": mean_bias_prob,
            "std_bias_prob": std_bias_prob,
            
            # Loss components
            "ce_loss": ce_loss.mean().item(),
            "weighted_loss": loss.item(),
        }
        
        return loss, metadata
        # """
        # Compute the reweighted loss with annealing using full probability distributions.
        
        # This is the proper implementation that applies annealing to the full distribution.
        
        # Args:
        #     logits: Model logits [batch_size, num_labels]
        #     labels: Ground truth labels [batch_size]
        #     bias_probs_full: Full probability distributions from shallow model [batch_size, num_labels]
        #     current_step: Current training step
        #     total_steps: Total number of training steps
            
        # Returns:
        #     Tuple of (loss, dict with metadata)
        # """
        # # Ensure inputs are on correct device
        # logits = logits.to(self.device)
        # labels = labels.to(self.device)
        # bias_probs_full = bias_probs_full.to(self.device)

        # # Compute annealing factor (varies from 1 to annealing_min)
        # alpha = self.compute_annealing_factor(
        #     current_step=current_step,
        #     total_steps=total_steps,
        #     a=self.selfdebias_config.annealing_min
        # )
        
        # with torch.no_grad():
        #     # Safely anneal the probability distributions (clamp for stability)
        #     annealed_probs = self.anneal_probability_distribution(
        #         bias_probs_full, 
        #         alpha,
        #         eps=1e-12  # Prevent numerical instability
        #     )
            
        #     # Extract probability for correct label
        #     bias_prob_correct = annealed_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            
        #     # Compute weights and detach from computation graph
        #     weights = (1.0 - bias_prob_correct).detach()
        #     weights = weights.clamp(min=0.0, max=1.0)  # Safety clamp
        
        # # Compute cross entropy loss
        # ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # # Apply weights and normalize
        # weighted_loss = weights * ce_loss
        # loss = weighted_loss.sum() / weights.sum().clamp(min=1e-8)  # Normalize by weight sum
        
        # # Compute metadata
        # with torch.no_grad():
        #     mean_weight = weights.mean().item()
        #     mean_bias_prob = bias_prob_correct.mean().item()
        #     alpha_current = alpha  # Track current annealing value
        
        # metadata = {
        #     # Training dynamics
        #     "alpha": alpha_current,
        #     "step": current_step,
        #     "total_steps": total_steps,
            
        #     # Weight statistics  
        #     "mean_weight": mean_weight,
            
        #     # Bias probability statistics
        #     "mean_bias_prob": mean_bias_prob,
            
        #     # Loss components
        #     "ce_loss": ce_loss.mean().item(),
        #     "weighted_loss": loss.item(),
        # }
        
        # return loss, metadata
    
    def _save_bias_distributions(
        self,
        prob_distributions: torch.Tensor,
        labels: torch.Tensor,
        cache_key: str
    ):
        """Save full probability distributions to disk."""
        cache_path = self.cache_dir / f"{cache_key}.pt"
        torch.save({
            'prob_distributions': prob_distributions,
            'labels': labels
        }, cache_path)
        logger.info(f"Saved bias distributions to {cache_path}")
    
    def _load_bias_distributions(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load full probability distributions from disk."""
        cache_path = self.cache_dir / f"{cache_key}.pt"
        if cache_path.exists():
            data = torch.load(cache_path)
            logger.info(f"Loaded bias distributions from {cache_path}")
            return data['prob_distributions'], data['labels']
        return None

