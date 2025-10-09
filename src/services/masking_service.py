"""
Masking service for MaskTune implementation.

This service handles the core logic of MaskTune:
- Saliency map calculation using L2 norm of gradients
- Input masking based on saliency scores
- Dataset transformation to create masked datasets
- Debug visualization of masking process
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


class MaskingService:
    """
    Service for computing saliency maps and applying masking to inputs.
    
    This implements the MaskTune methodology for identifying and masking
    discriminative features to mitigate spurious correlations.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the MaskingService.
        
        Args:
            model: The trained model to compute saliency scores with
            tokenizer: The tokenizer used for the model
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Special tokens that should never be masked
        self.protected_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        # Remove None values in case some special tokens are not defined
        self.protected_tokens = {token_id for token_id in self.protected_tokens if token_id is not None}
        
        # Debug information storage
        self.debug_samples: List[Dict[str, Any]] = []
        
    def compute_saliency_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        method: str = "grad_l2"
    ) -> torch.Tensor:
        """
        Compute saliency scores for input tokens using gradient-based methods.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size]
            method: Saliency computation method ("grad_l2")
            
        Returns:
            Saliency scores for each token [batch_size, seq_len]
        """
        if method != "grad_l2":
            raise ValueError(f"Unsupported saliency method: {method}")
            
        self.model.eval()
        
        # Move tensors to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Clear any existing gradients
        self.model.zero_grad()
        
        # Get input embeddings and enable gradient tracking
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds.requires_grad_(True)
        
        # Single forward pass for the entire batch
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get logits
        logits = outputs.logits  # [batch_size, num_classes]
        
        # Get predicted classes for each sample
        predicted_classes = torch.argmax(logits, dim=1)  # [batch_size]
        
        # Create one-hot encoding for the predicted classes
        batch_size, num_classes = logits.shape
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(batch_size), predicted_classes] = 1.0
        
        # Compute gradients w.r.t. predicted classes
        # This is equivalent to taking the gradient of each predicted class logit
        loss = torch.sum(logits * one_hot)
        
        # Backward pass
        loss.backward()
        
        # Compute L2 norm of gradients for each token
        if inputs_embeds.grad is not None:
            # inputs_embeds.grad: [batch_size, seq_len, hidden_size]
            token_gradients = inputs_embeds.grad  
            # Compute L2 norm across the hidden dimension for each token
            saliency_scores = torch.norm(token_gradients, p=2, dim=2)  # [batch_size, seq_len]
        else:
            saliency_scores = torch.zeros_like(input_ids, dtype=torch.float)
        
        return saliency_scores.detach()

    def compute_saliency_scores_batched(
        self,
        dataset: Dataset,
        text_columns: List[str],
        label_column: str = "label",
        batch_size: int = 32,
        max_length: int = 512
    ) -> List[torch.Tensor]:
        """
        Compute saliency scores for an entire dataset in an efficient batched manner.
        
        Args:
            dataset: Dataset to compute saliency for
            text_columns: List of text column names to process
            label_column: Name of the label column
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            List of saliency score tensors, one per sample
        """
        logger.info(f"Computing saliency scores for {len(dataset)} samples")
        
        all_saliency_scores = []
        
        # Prepare tokenization function
        def tokenize_batch(batch):
            if len(text_columns) == 1:
                texts = batch[text_columns[0]]
            else:
                texts = []
                for i in range(len(batch[text_columns[0]])):
                    combined_text = " ".join([
                        str(batch[col][i]) for col in text_columns
                    ])
                    texts.append(combined_text)
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Computing saliency"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Tokenize batch
            tokenized = tokenize_batch(batch)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Get labels
            if isinstance(batch[label_column], list):
                labels = torch.tensor(batch[label_column])
            else:
                labels = torch.tensor([batch[label_column]])
            
            # Compute saliency scores for this batch
            batch_saliency = self.compute_saliency_scores(
                input_ids, attention_mask, labels
            )
            
            # Store individual saliency tensors
            for j in range(batch_saliency.size(0)):
                all_saliency_scores.append(batch_saliency[j].cpu())
        
        return all_saliency_scores
    
    def identify_tokens_to_mask(
        self,
        saliency_scores: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: str = "threshold",
        threshold_multiplier: float = 2.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Identify tokens to mask based on saliency scores.
        
        Args:
            saliency_scores: Saliency scores for each token [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            strategy: Masking strategy ("threshold" or "top_k")
            threshold_multiplier: Multiplier for mean + std threshold (default: 2.0)
            top_k: Number of top tokens to mask (only for "top_k" strategy)
            
        Returns:
            Boolean mask indicating which tokens to mask [batch_size, seq_len]
        """
        batch_size, seq_len = saliency_scores.shape
        mask_indices = torch.zeros_like(saliency_scores, dtype=torch.bool)
        
        for i in range(batch_size):
            # Get valid tokens (non-padding, non-special)
            valid_mask = (attention_mask[i] == 1)
            for protected_token in self.protected_tokens:
                valid_mask &= (input_ids[i] != protected_token)
            
            if not valid_mask.any():
                continue
                
            valid_scores = saliency_scores[i][valid_mask]
            
            if strategy == "threshold":
                # Use mean + threshold_multiplier * std as threshold
                mean_score = valid_scores.mean(dim=-1)
                std_score = valid_scores.std(dim=-1)
                threshold = mean_score + threshold_multiplier * std_score
                
                # Mark tokens above threshold for masking
                high_saliency = (saliency_scores[i] > threshold).cpu() & valid_mask
                mask_indices[i] = high_saliency
                
            elif strategy == "top_k":
                if top_k is None:
                    raise ValueError("top_k must be specified (int > 0 or float in (0,1]].")

                # Get indices of valid tokens
                valid_indices = torch.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    continue

                n_valid = len(valid_indices)

                # Compute k from top_k (support int or fraction)
                if isinstance(top_k, int):
                    if top_k <= 0:
                        raise ValueError("If top_k is int, it must be > 0.")
                    k = top_k
                elif isinstance(top_k, float):
                    if not (0.0 < top_k <= 1.0):
                        raise ValueError("If top_k is float, it must be in (0,1]. "
                                        "Example: 0.2 means 20% of valid tokens.")
                    k = int(math.ceil(n_valid * top_k))
                else:
                    raise TypeError("top_k must be int or float.")

                # Ensure k does not exceed the number of valid tokens
                k = min(k, n_valid)
                if k == 0:
                    k = 1

                # IMPORTANT: make sure valid_scores is filtered to valid_indices
                # If not already done, uncomment the next line:
                # valid_scores = scores[valid_indices]

                # Select top-k scores among valid tokens
                _, top_indices = torch.topk(valid_scores, k)

                # Map indices back to the original positions
                original_indices = valid_indices[top_indices]
                mask_indices[i][original_indices] = True

                
            else:
                raise ValueError(f"Unsupported masking strategy: {strategy}")
        
        return mask_indices
    
    def apply_masking(
        self,
        input_ids: torch.Tensor,
        mask_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply masking to input tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            mask_indices: Boolean mask indicating which tokens to mask
            
        Returns:
            Masked input token IDs
        """
        masked_input_ids = input_ids.clone()
        
        # Replace marked tokens with [MASK] token
        if self.tokenizer.mask_token_id is not None:
            masked_input_ids[mask_indices] = self.tokenizer.mask_token_id
        else:
            # Fallback to [UNK] if [MASK] is not available
            logger.warning("MASK token not available, using UNK token instead")
            masked_input_ids[mask_indices] = self.tokenizer.unk_token_id
            
        return masked_input_ids
    
    def create_debug_visualization(
        self,
        sample: Dict[str, Any],
        text_columns: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        saliency_scores: torch.Tensor,
        mask_indices: torch.Tensor,
        masked_input_ids: torch.Tensor,
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        Create a comprehensive debug visualization for a single sample.
        
        Args:
            sample: Original dataset sample
            text_columns: Text columns being processed
            input_ids: Original input token IDs
            attention_mask: Attention mask
            saliency_scores: Computed saliency scores
            mask_indices: Boolean mask of tokens to mask
            masked_input_ids: Input IDs after masking
            sample_idx: Index of the sample in the dataset
            
        Returns:
            Dictionary containing debug information and visualizations
        """
        # Get original and masked text
        original_text = " ".join([str(sample[col]) for col in text_columns])
        masked_text = self.tokenizer.decode(
            masked_input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Get tokens and their properties
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_ids = input_ids[0].tolist()
        scores = saliency_scores[0].tolist()
        is_masked = mask_indices[0].tolist()
        is_valid = attention_mask[0].tolist()
        
        # Create token-level analysis
        token_analysis = []
        num_masked = 0
        total_saliency_masked = 0.0
        total_saliency_unmasked = 0.0
        
        for i, (token, token_id, score, masked, valid) in enumerate(zip(
            tokens, token_ids, scores, is_masked, is_valid
        )):
            is_protected = token_id in self.protected_tokens
            
            analysis = {
                "position": i,
                "token": token,
                "token_id": token_id,
                "saliency_score": round(score, 4),
                "is_masked": masked,
                "is_valid": bool(valid),
                "is_protected": is_protected,
                "readable_token": self.tokenizer.convert_tokens_to_string([token]).strip()
            }
            
            if valid and not is_protected:
                if masked:
                    num_masked += 1
                    total_saliency_masked += score
                else:
                    total_saliency_unmasked += score
            
            token_analysis.append(analysis)
        
        # Compute statistics
        valid_scores = [s for s, v, p in zip(scores, is_valid, [tid in self.protected_tokens for tid in token_ids]) 
                       if v and not p]
        
        # Calculate valid tokens for masking (excluding protected tokens)
        valid_tokens_for_masking = sum(1 for i, (valid, tid) in enumerate(zip(is_valid, token_ids)) 
                                      if valid and tid not in self.protected_tokens)
        
        # Add debug logging for the first few samples
        if sample_idx < 3:
            logger.info(f"Debug sample {sample_idx} token analysis:")
            logger.info(f"  Total tokens: {len(tokens)}")
            logger.info(f"  Valid tokens (non-padding): {sum(is_valid)}")
            logger.info(f"  Protected tokens: {sum(1 for tid in token_ids if tid in self.protected_tokens)}")
            logger.info(f"  Valid tokens for masking: {valid_tokens_for_masking}")
            logger.info(f"  Tokens masked: {num_masked}")
            logger.info(f"  Masking percentage: {round((num_masked / max(1, valid_tokens_for_masking)) * 100, 2)}%")
        
        stats = {
            "total_tokens": len(tokens),
            "valid_tokens": sum(is_valid),
            "protected_tokens": sum(1 for tid in token_ids if tid in self.protected_tokens),
            "tokens_masked": num_masked,
            "valid_tokens_for_masking": valid_tokens_for_masking,
            "masking_percentage": round((num_masked / max(1, valid_tokens_for_masking)) * 100, 2),
            "saliency_stats": {
                "mean": round(np.mean(valid_scores), 4) if valid_scores else 0.0,
                "std": round(np.std(valid_scores), 4) if valid_scores else 0.0,
                "min": round(min(valid_scores), 4) if valid_scores else 0.0,
                "max": round(max(valid_scores), 4) if valid_scores else 0.0,
                "median": round(np.median(valid_scores), 4) if valid_scores else 0.0
            },
            "avg_saliency_masked": round(total_saliency_masked / max(1, num_masked), 4),
            "avg_saliency_unmasked": round(total_saliency_unmasked / max(1, len(valid_scores) - num_masked), 4)
        }
        
        # Create text comparison with highlighting
        highlighted_comparison = self._create_text_highlighting(
            original_text, masked_text, tokens, is_masked, is_valid, scores
        )
        
        return {
            "sample_index": sample_idx,
            "label": sample.get("label", sample.get("labels", "unknown")),
            "original_text": original_text,
            "masked_text": masked_text,
            "statistics": stats,
            "token_analysis": token_analysis,
            "text_comparison": highlighted_comparison,
            "masking_summary": f"Masked {num_masked} out of {valid_tokens_for_masking} valid tokens ({stats['masking_percentage']}%)"
        }
    
    def _create_text_highlighting(
        self,
        original_text: str,
        masked_text: str,
        tokens: List[str],
        is_masked: List[bool],
        is_valid: List[bool],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        Create a text highlighting visualization showing which parts were masked.
        
        Returns:
            Dictionary with highlighted text and legend
        """
        # Create highlighted version showing masked tokens
        highlighted_tokens = []
        
        for token, masked, valid, score in zip(tokens, is_masked, is_valid, scores):
            readable_token = self.tokenizer.convert_tokens_to_string([token]).strip()
            
            if not valid or token.startswith('[') and token.endswith(']'):
                # Skip special tokens in visualization
                continue
                
            if masked:
                highlighted_tokens.append(f"[MASKED:{readable_token}({score:.2f})]")
            else:
                highlighted_tokens.append(f"{readable_token}({score:.2f})")
        
        return {
            "highlighted_text": " ".join(highlighted_tokens),
            "legend": {
                "**[MASKED:token]**": "Token was masked due to high saliency",
                "normal_text": "Token was preserved"
            },
            "visualization_note": "This shows which tokens were identified as salient and masked"
        }
    
    def save_debug_visualizations(
        self,
        output_dir: Path,
        strategy: str,
        threshold_multiplier: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> None:
        """
        Save debug visualizations to files.
        
        Args:
            output_dir: Directory to save visualizations
            strategy: Masking strategy used
            threshold_multiplier: Threshold multiplier (if applicable)
            top_k: Top-k value (if applicable)
        """
        if not self.debug_samples:
            logger.warning("No debug samples to save")
            return
        
        debug_dir = output_dir / "masking_debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Save detailed debug samples
        debug_file = debug_dir / "masking_debug_samples.json"
        with open(debug_file, 'w') as f:
            json.dump(self.debug_samples, f, indent=2)
        
        # Create summary statistics
        summary = self._create_debug_summary(strategy, threshold_multiplier, top_k)
        summary_file = debug_dir / "masking_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create human-readable report
        report_file = debug_dir / "masking_report.txt"
        self._create_human_readable_report(report_file, strategy, threshold_multiplier, top_k)
        
        logger.info(f"Saved masking debug visualizations to {debug_dir}")
        logger.info(f"- Detailed samples: {debug_file}")
        logger.info(f"- Summary statistics: {summary_file}")
        logger.info(f"- Human-readable report: {report_file}")
    
    def _create_debug_summary(
        self,
        strategy: str,
        threshold_multiplier: Optional[float],
        top_k: Optional[int]
    ) -> Dict[str, Any]:
        """Create summary statistics from debug samples."""
        if not self.debug_samples:
            return {}
        
        masking_percentages = [s["statistics"]["masking_percentage"] for s in self.debug_samples]
        avg_saliency_masked = [s["statistics"]["avg_saliency_masked"] for s in self.debug_samples]
        avg_saliency_unmasked = [s["statistics"]["avg_saliency_unmasked"] for s in self.debug_samples]
        
        return {
            "strategy": strategy,
            "strategy_params": {
                "threshold_multiplier": threshold_multiplier,
                "top_k": top_k
            },
            "total_samples_analyzed": len(self.debug_samples),
            "masking_statistics": {
                "avg_masking_percentage": round(np.mean(masking_percentages), 2),
                "std_masking_percentage": round(np.std(masking_percentages), 2),
                "min_masking_percentage": round(min(masking_percentages), 2),
                "max_masking_percentage": round(max(masking_percentages), 2),
                "median_masking_percentage": round(np.median(masking_percentages), 2)
            },
            "saliency_analysis": {
                "avg_saliency_of_masked_tokens": round(np.mean(avg_saliency_masked), 4),
                "avg_saliency_of_unmasked_tokens": round(np.mean(avg_saliency_unmasked), 4),
                "saliency_ratio_masked_vs_unmasked": round(np.mean(avg_saliency_masked) / max(0.0001, np.mean(avg_saliency_unmasked)), 2)
            },
            # "samples": [
            #     {
            #         "index": s["sample_index"],
            #         "label": s["label"],
            #         "masking_percentage": s["statistics"]["masking_percentage"],
            #         "masking_summary": s["masking_summary"]
            #     }
                # for s in self.debug_samples
            # ]
        }
    
    def _create_human_readable_report(
        self,
        report_file: Path,
        strategy: str,
        threshold_multiplier: Optional[float],
        top_k: Optional[int]
    ) -> None:
        """Create a human-readable text report of the masking process."""
        with open(report_file, 'w') as f:
            f.write("MaskTune Masking Debug Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Strategy: {strategy}\n")
            if threshold_multiplier is not None:
                f.write(f"Threshold Multiplier: {threshold_multiplier}\n")
            if top_k is not None:
                f.write(f"Top-K: {top_k}\n")
            f.write(f"Samples Analyzed: {len(self.debug_samples)}\n\n")
            
            # Summary statistics
            summary = self._create_debug_summary(strategy, threshold_multiplier, top_k)
            if summary:
                f.write("Summary Statistics:\n")
                f.write(f"- Average masking percentage: {summary['masking_statistics']['avg_masking_percentage']}%\n")
                f.write(f"- Masking percentage range: {summary['masking_statistics']['min_masking_percentage']}% - {summary['masking_statistics']['max_masking_percentage']}%\n")
                f.write(f"- Saliency ratio (masked/unmasked): {summary['saliency_analysis']['saliency_ratio_masked_vs_unmasked']}\n\n")
            
            # Individual samples
            f.write("Individual Sample Analysis:\n")
            f.write("-" * 30 + "\n\n")
            
            for sample in self.debug_samples:
                f.write(f"Sample {sample['sample_index']} (Label: {sample['label']}):\n")
                f.write(f"Original: {sample['original_text'][:100]}{'...' if len(sample['original_text']) > 100 else ''}\n")
                f.write(f"Masked:   {sample['masked_text'][:100]}{'...' if len(sample['masked_text']) > 100 else ''}\n")
                f.write(f"Masking:  {sample['masking_summary']}\n")
                f.write(f"Highlighting: {sample['text_comparison']['highlighted_text'][:150]}{'...' if len(sample['text_comparison']['highlighted_text']) > 150 else ''}\n")
                f.write("\n")
        
    def create_masked_dataset(
        self,
        dataset: Dataset,
        text_columns: List[str],
        label_column: str = "label",
        batch_size: int = 32,
        masking_strategy: str = "threshold",
        threshold_multiplier: float = 2.0,
        top_k: Optional[int] = None,
        max_length: int = 512,
        extract_debug_samples: bool = True,
        num_debug_samples: int = 10,
        save_debug_visualizations: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dataset:
        """
        Create a masked version of the dataset using saliency-based masking.
        
        Args:
            dataset: Input dataset to mask
            text_columns: List of text column names to process
            label_column: Name of the label column
            batch_size: Batch size for processing (increased default for efficiency)
            masking_strategy: Strategy for selecting tokens to mask
            threshold_multiplier: Multiplier for threshold strategy
            top_k: Number of top tokens to mask for top_k strategy
            max_length: Maximum sequence length
            extract_debug_samples: Whether to extract debug samples
            num_debug_samples: Number of debug samples to extract
            save_debug_visualizations: Whether to save debug visualizations
            output_dir: Output directory for saving debug visualizations
            
        Returns:
            New dataset with masked text columns
        """
        logger.info(f"Creating masked dataset with {len(dataset)} samples")
        
        # Clear previous debug samples
        self.debug_samples = []
        
        # Step 1: Compute all saliency scores efficiently
        all_saliency_scores = self.compute_saliency_scores_batched(
            dataset, text_columns, label_column, batch_size, max_length
        )
        
        # Step 2: Process dataset efficiently with precomputed saliency scores
        masked_data = []
        debug_sample_indices = []
        
        # Select indices for debug samples (evenly distributed)
        if extract_debug_samples and num_debug_samples > 0:
            step = max(1, len(dataset) // num_debug_samples)
            debug_sample_indices = list(range(0, len(dataset), step))[:num_debug_samples]
            logger.info(f"Will collect debug samples for indices: {debug_sample_indices}")
        
        def tokenize_single(text_list):
            if len(text_columns) == 1:
                texts = text_list
            else:
                texts = [" ".join([str(text_list[i]) for i in range(len(text_columns))])]
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length", 
                max_length=max_length,
                return_tensors="pt"
            )
        
        logger.info("Applying masking to samples")
        for i in tqdm(range(len(dataset)), desc="Masking samples"):
            sample = dataset[i]
            saliency_scores = all_saliency_scores[i].unsqueeze(0)  # Add batch dimension
            
            # Tokenize current sample
            if len(text_columns) == 1:
                texts = [sample[text_columns[0]]]
            else:
                combined_text = " ".join([str(sample[col]) for col in text_columns])
                texts = [combined_text]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Identify tokens to mask
            mask_indices = self.identify_tokens_to_mask(
                saliency_scores,
                input_ids,
                attention_mask,
                strategy=masking_strategy,
                threshold_multiplier=threshold_multiplier,
                top_k=top_k
            )
            
            # Apply masking
            masked_input_ids = self.apply_masking(input_ids, mask_indices)
            
            # Collect debug information if this sample is selected
            if extract_debug_samples and i in debug_sample_indices:
                debug_info = self.create_debug_visualization(
                    sample=sample,
                    text_columns=text_columns,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    saliency_scores=saliency_scores,
                    mask_indices=mask_indices,
                    masked_input_ids=masked_input_ids,
                    sample_idx=i
                )
                self.debug_samples.append(debug_info)
                
                # Log debug info for the first few samples
                if len(self.debug_samples) <= 3:
                    logger.info(f"Debug sample {i}: {debug_info['masking_summary']}")
                    logger.info(f"  Original: {debug_info['original_text'][:100]}...")
                    logger.info(f"  Masked:   {debug_info['masked_text'][:100]}...")
            
            # Decode the masked tokens
            masked_text = self.tokenizer.decode(
                masked_input_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Create new sample with masked text
            new_sample = dict(sample)
            new_sample[text_columns[0]] = masked_text
            
            masked_data.append(new_sample)
        
        # Save debug visualizations if requested
        if save_debug_visualizations and self.debug_samples and output_dir:
            self.save_debug_visualizations(
                output_dir=output_dir,
                strategy=masking_strategy,
                threshold_multiplier=threshold_multiplier,
                top_k=top_k
            )
        
        # Create new dataset
        masked_dataset = Dataset.from_list(masked_data)
        
        logger.info(f"Created masked dataset with {len(masked_dataset)} samples")
        if self.debug_samples:
            avg_masking = np.mean([s["statistics"]["masking_percentage"] for s in self.debug_samples])
            logger.info(f"Average masking percentage from debug samples: {avg_masking:.2f}%")
        
        return masked_dataset 