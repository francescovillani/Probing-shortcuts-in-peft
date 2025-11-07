import evaluate
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Centralized service for model evaluation operations.
    
    This service encapsulates all evaluation logic including:
    - Basic model evaluation with metrics
    - Confidence metrics computation for backdoor analysis
    - Embedding similarity analysis between clean/triggered examples
    """
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def _get_model_device(self, model) -> torch.device:
        """
        Resolve the device the model is actually on. Falls back to service device.
        """
        try:
            return next(model.parameters()).device
        except StopIteration:
            return self.device
    
    def _move_batch_to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Recursively move a batch dict to the specified device, leaving non-tensors as-is.
        """
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        
    def execute(
        self,
        model,
        dataloader,
        is_hans: bool = False,
        desc: str = "Evaluating",
        metrics: List[str] = None,
        save_predictions: bool = False,
        compute_confidence: bool = False,
        confidence_target_label: Optional[Union[int, List[int]]] = None,
        compute_hidden_similarities: bool = False,
        dataset_service = None,
        dataset_config = None,
    ) -> Dict[str, Any]:
        """
        Execute a comprehensive evaluation on the given model and dataloader.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader with evaluation data
            is_hans: Whether this is HANS evaluation (special logit processing)
            desc: Description for progress bars
            metrics: List of metrics to compute (default: ["accuracy"])
            save_predictions: Whether to include predictions and labels in results
            compute_confidence: Whether to compute confidence metrics
            confidence_target_label: Target label(s) for confidence analysis
            compute_hidden_similarities: Whether to compute hidden similarities between clean/triggered examples
            dataset_service: DatasetService instance (required if compute_hidden_similarities=True)
            dataset_config: DatasetConfig instance (required if compute_hidden_similarities=True)
            
        Returns:
            Dictionary containing all evaluation results
        """
        if metrics is None:
            metrics = ["accuracy"]
        
        # Validate hidden similarities requirements
        if compute_hidden_similarities:
            if dataset_service is None:
                raise ValueError("dataset_service is required when compute_hidden_similarities=True")
            if dataset_config is None:
                raise ValueError("dataset_config is required when compute_hidden_similarities=True")
            if not dataset_config.poisoning or not dataset_config.poisoning.enabled:
                logger.warning("compute_hidden_similarities=True but dataset has no poisoning config. Skipping similarity computation.")
                compute_hidden_similarities = False
        
        # Basic evaluation
        results = self._evaluate_model(
            model=model,
            dataloader=dataloader,
            device=self.device,
            is_hans=is_hans,
            desc=desc,
            metrics=metrics,
            save_predictions=save_predictions,
            compute_confidence=compute_confidence,
            confidence_target_label=confidence_target_label,
        )
        
        # Compute hidden similarities if requested
        if compute_hidden_similarities:
            try:
                logger.info("Computing hidden similarities between clean and triggered examples...")
                
                # Create clean and triggered dataloaders using the dataset service
                clean_dataloader, triggered_dataloader = dataset_service.create_clean_triggered_dataloaders(
                    config=dataset_config,
                    text_field=dataset_config.text_field,
                    label_field=dataset_config.label_field
                )
                
                similarity_results = self.compute_hidden_similarities(
                    model=model,
                    clean_dataloader=clean_dataloader,
                    triggered_dataloader=triggered_dataloader,
                    desc=f"{desc} - Hidden Similarities"
                )
                
                results["hidden_similarities"] = similarity_results
                
            except Exception as e:
                logger.error(f"Failed to compute hidden similarities: {e}")
                results["hidden_similarities"] = {"error": str(e)}
            
        return results
    
    def compute_hidden_similarities(
        self,
        model,
        clean_dataloader,
        triggered_dataloader,
        desc: str = "Computing hidden similarities",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute cosine similarities between clean and triggered versions of examples
        using hidden states before the classification head.
        
        Args:
            model: The trained model
            clean_dataloader: DataLoader with clean examples
            triggered_dataloader: DataLoader with triggered examples (same order as clean)
            desc: Description for progress bar
            max_samples: Optional limit on number of samples to process (for efficiency)
        
        Returns:
            Dictionary with similarity metrics for hidden states
        """
        # Input validation
        if len(clean_dataloader) != len(triggered_dataloader):
            logger.warning(f"DataLoader lengths differ: clean={len(clean_dataloader)}, "
                          f"triggered={len(triggered_dataloader)}. Will process min length.")
        
        model.eval()
        hidden_similarities = []
        samples_processed = 0
        
        logger.info("Computing hidden state similarities between clean and triggered examples...")
        
        with torch.no_grad():
            model_device = self._get_model_device(model)
            # Zip the dataloaders to process corresponding clean and triggered examples together
            paired_batches = zip(clean_dataloader, triggered_dataloader)
            
            for clean_batch, triggered_batch in tqdm(paired_batches, desc=desc):
                # Check max_samples limit
                if max_samples is not None and samples_processed >= max_samples:
                    logger.info(f"Reached max_samples limit of {max_samples}, stopping.")
                    break
                
                # Move batches to the actual model device
                clean_batch = self._move_batch_to_device(clean_batch, model_device)
                triggered_batch = self._move_batch_to_device(triggered_batch, model_device)
                
                # Ensure batch sizes match
                batch_size = min(clean_batch['input_ids'].size(0), triggered_batch['input_ids'].size(0))
                if clean_batch['input_ids'].size(0) != triggered_batch['input_ids'].size(0):
                    logger.warning(f"Batch sizes differ, using min size: {batch_size}")
                    for key in clean_batch:
                        clean_batch[key] = clean_batch[key][:batch_size]
                    for key in triggered_batch:
                        triggered_batch[key] = triggered_batch[key][:batch_size]
                
                try:
                    # Extract the backbone for proper model type detection
                    backbone = self._get_model_backbone(model)
                    
                    # Extract hidden states for both clean and triggered examples
                    _, clean_hidden = self._extract_embeddings_and_hidden_states(model, clean_batch)
                    _, triggered_hidden = self._extract_embeddings_and_hidden_states(model, triggered_batch)
                    
                    # Use CLS token representation from last hidden state
                    # Handle different model architectures for CLS token position
                    clean_hidden_cls = self._extract_cls_representation(clean_hidden, clean_batch, backbone)
                    triggered_hidden_cls = self._extract_cls_representation(triggered_hidden, triggered_batch, backbone)
                    
                    # Compute cosine similarities in batch
                    batch_similarities = F.cosine_similarity(clean_hidden_cls, triggered_hidden_cls, dim=1)
                    hidden_similarities.extend(batch_similarities.cpu().tolist())
                    
                    samples_processed += batch_size
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
        
        if not hidden_similarities:
            logger.error("No similarities computed - check model compatibility and data")
            return {"error": "No similarities computed"}
        
        # Compute comprehensive statistics
        similarities_tensor = torch.tensor(hidden_similarities)
        
        results = {
            "hidden_similarities": {
                "mean": float(similarities_tensor.mean()),
                "std": float(similarities_tensor.std()),
                "min": float(similarities_tensor.min()),
                "max": float(similarities_tensor.max()),
                "median": float(similarities_tensor.median()),
                "q25": float(similarities_tensor.quantile(0.25)),
                "q75": float(similarities_tensor.quantile(0.75)),
                "count": len(hidden_similarities),
                "samples_processed": samples_processed
            }
        }
        
        logger.info(f"Hidden similarities - Mean: {results['hidden_similarities']['mean']:.4f}, "
                   f"Std: {results['hidden_similarities']['std']:.4f}, "
                   f"Samples: {samples_processed}")
        
        return results

    def _get_model_backbone(self, model):
        """
        Extract the actual backbone model from potentially nested PEFT models.
        
        Args:
            model: The model (potentially wrapped in PEFT)
            
        Returns:
            The backbone transformer model
        """
        # Get model's base architecture (handles PEFT models)
        if hasattr(model, 'base_model'):
            # For PEFT models, get the base model
            base_model = model.base_model
            if hasattr(base_model, 'model'):
                # For some PEFT models, need to go deeper
                backbone = base_model.model
            else:
                backbone = base_model
        else:
            # For regular models
            backbone = model
        
        # Additional layer for complex PEFT hierarchies
        # Some models have nested structures like PeftModel -> Model -> TransformerModel
        if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'model'):
            backbone = backbone.base_model.model
        elif hasattr(backbone, 'base_model'):
            backbone = backbone.base_model
            
        return backbone

    def compute_confidence_metrics(
        self,
        model,
        dataloader,
        target_label: Union[int, List[int]],
        desc: str = "Computing confidence metrics",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute confidence scores and logit differences for backdoor strength analysis.
        
        Optimized version with vectorized operations and better memory management.
        
        Args:
            model: The trained model
            dataloader: DataLoader with examples to analyze
            target_label: Target label(s) for backdoor. Can be int or list of ints
            desc: Description for progress bar
            max_samples: Optional limit on number of samples to process
        
        Returns:
            Dictionary with confidence metrics:
            - target_confidence: Statistics about confidence in target label
            - logit_differences: Statistics about target-true logit differences
            - prediction_distribution: How predictions are distributed
        """
        # Input validation
        if isinstance(target_label, int):
            target_labels = [target_label]
        elif isinstance(target_label, list):
            target_labels = target_label
        else:
            raise ValueError(f"target_label must be int or list of ints, got {type(target_label)}")
        
        if not target_labels:
            raise ValueError("target_labels cannot be empty")
        
        model.eval()
        model_device = self._get_model_device(model)
        
        # Accumulate results in batches for memory efficiency
        all_target_confidences = []
        all_logit_differences = []
        target_predictions = 0
        total_samples = 0
        
        logger.info(f"Computing confidence metrics for target label(s): {target_labels}")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Check max_samples limit
                if max_samples is not None and total_samples >= max_samples:
                    logger.info(f"Reached max_samples limit of {max_samples}, stopping.")
                    break
                
                batch = self._move_batch_to_device(batch, model_device)
                
                try:
                    outputs = model(**batch)
                    logits = outputs.logits  # [batch_size, num_classes]
                    true_labels = batch["labels"]  # [batch_size]
                    batch_size = logits.size(0)
                    
                    # Limit batch size if needed
                    if max_samples is not None:
                        remaining_samples = max_samples - total_samples
                        if remaining_samples < batch_size:
                            batch_size = remaining_samples
                            logits = logits[:batch_size]
                            true_labels = true_labels[:batch_size]
                    
                    # Compute softmax probabilities for the entire batch
                    probabilities = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
                    
                    # Vectorized target determination
                    # For each sample, determine the appropriate target label
                    target_indices = torch.zeros(batch_size, dtype=torch.long, device=model_device)
                    
                    for i, true_label in enumerate(true_labels):
                        if true_label.item() in target_labels:
                            # If true label is in target_labels, use it as target
                            target_indices[i] = true_label
                        else:
                            # For cross-label attacks, use the first target label
                            target_indices[i] = target_labels[0]
                    
                    # Vectorized confidence computation
                    batch_target_confidences = probabilities[torch.arange(batch_size), target_indices]
                    all_target_confidences.extend(batch_target_confidences.cpu().tolist())
                    
                    # Vectorized logit difference computation
                    target_logits = logits[torch.arange(batch_size), target_indices]
                    true_logits = logits[torch.arange(batch_size), true_labels]
                    batch_logit_differences = target_logits - true_logits
                    all_logit_differences.extend(batch_logit_differences.cpu().tolist())
                    
                    # Vectorized prediction counting
                    predicted_labels = torch.argmax(logits, dim=-1)
                    target_matches = (predicted_labels == target_indices).sum().item()
                    target_predictions += target_matches
                    
                    total_samples += batch_size
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
        
        if not all_target_confidences:
            logger.error("No confidence metrics computed - check model and data compatibility")
            return {"error": "No confidence metrics computed"}
        
        # Compute comprehensive statistics
        target_conf_tensor = torch.tensor(all_target_confidences)
        logit_diff_tensor = torch.tensor(all_logit_differences)
        
        results = {
            "target_confidence": {
                "mean": float(target_conf_tensor.mean()),
                "std": float(target_conf_tensor.std()),
                "min": float(target_conf_tensor.min()),
                "max": float(target_conf_tensor.max()),
                "median": float(target_conf_tensor.median()),
                "q25": float(target_conf_tensor.quantile(0.25)),
                "q75": float(target_conf_tensor.quantile(0.75)),
                # Additional percentiles for better analysis
                "q10": float(target_conf_tensor.quantile(0.10)),
                "q90": float(target_conf_tensor.quantile(0.90)),
                "count": len(all_target_confidences)
            },
            "logit_differences": {
                "mean": float(logit_diff_tensor.mean()),
                "std": float(logit_diff_tensor.std()),
                "min": float(logit_diff_tensor.min()),
                "max": float(logit_diff_tensor.max()),
                "median": float(logit_diff_tensor.median()),
                "q25": float(logit_diff_tensor.quantile(0.25)),
                "q75": float(logit_diff_tensor.quantile(0.75)),
                "q10": float(logit_diff_tensor.quantile(0.10)),
                "q90": float(logit_diff_tensor.quantile(0.90)),
                "count": len(all_logit_differences)
            },
            "prediction_stats": {
                "target_prediction_rate": target_predictions / total_samples if total_samples > 0 else 0.0,
                "total_samples": total_samples,
                "target_predictions": target_predictions,
                "attack_success_rate": target_predictions / total_samples if total_samples > 0 else 0.0  # Alias for clarity
            },
            "metadata": {
                "target_labels": target_labels,
                "samples_processed": total_samples
            }
        }
        
        # Enhanced logging
        logger.info(f"Confidence Metrics Summary:")
        logger.info(f"  Target Confidence - Mean: {results['target_confidence']['mean']:.4f}, "
                   f"Std: {results['target_confidence']['std']:.4f}, "
                   f"Median: {results['target_confidence']['median']:.4f}")
        logger.info(f"  Logit Differences - Mean: {results['logit_differences']['mean']:.4f}, "
                   f"Std: {results['logit_differences']['std']:.4f}, "
                   f"Median: {results['logit_differences']['median']:.4f}")
        logger.info(f"  Attack Success Rate: {results['prediction_stats']['attack_success_rate']:.4f}")
        logger.info(f"  Samples Processed: {total_samples}")
        
        return results

    def _extract_embeddings_and_hidden_states(self, model, batch):
        """
        Extract embeddings and hidden states from the model.
        
        Args:
            model: The model to extract representations from
            batch: Input batch with input_ids, attention_mask, etc.
        
        Returns:
            Tuple of (embeddings, hidden_states)
            - embeddings: Token embeddings after the embedding layer [batch_size, seq_len, hidden_dim]
            - hidden_states: Hidden states before classification head [batch_size, seq_len, hidden_dim]
        """
        # Get model's base architecture (handles PEFT models)
        if hasattr(model, 'base_model'):
            # For PEFT models, get the base model
            base_model = model.base_model
            if hasattr(base_model, 'model'):
                # For some PEFT models, need to go deeper
                backbone = base_model.model
            else:
                backbone = base_model
        else:
            # For regular models
            backbone = model
        
        # Additional layer for complex PEFT hierarchies
        # Some models have nested structures like PeftModel -> Model -> TransformerModel
        if hasattr(backbone, 'base_model') and hasattr(backbone.base_model, 'model'):
            backbone = backbone.base_model.model
        elif hasattr(backbone, 'base_model'):
            backbone = backbone.base_model
        
        # For direct RoBERTa model instances (when backbone is the RobertaModel itself)
        if hasattr(backbone, 'embeddings') and hasattr(backbone, 'encoder') and hasattr(backbone, 'pooler') and 'roberta' in str(type(backbone)).lower():
            # Direct RoBERTa architecture (e.g., when using base RobertaModel)
            embeddings = backbone.embeddings(input_ids=batch['input_ids'])
            
            # Get hidden states from the RoBERTa model
            roberta_outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = roberta_outputs.last_hidden_state
            
        # For RoBERTa-style models (wrapped)
        elif hasattr(backbone, 'roberta'):
            # Get embeddings from the embedding layer
            embeddings = backbone.roberta.embeddings(input_ids=batch['input_ids'])
            
            # Use the full RoBERTa model forward pass to get hidden states
            # This ensures proper attention mask handling
            roberta_outputs = backbone.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = roberta_outputs.last_hidden_state
            
        # For direct BERT model instances (when backbone is the BertModel itself)
        elif hasattr(backbone, 'embeddings') and hasattr(backbone, 'encoder') and hasattr(backbone, 'pooler') and 'bert' in str(type(backbone)).lower():
            # Direct BERT architecture (e.g., when using base BertModel)
            embeddings = backbone.embeddings(
                input_ids=batch['input_ids'],
                token_type_ids=batch.get('token_type_ids')
            )
            
            # Get hidden states from the BERT model
            bert_outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                token_type_ids=batch.get('token_type_ids'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = bert_outputs.last_hidden_state
            
        # For BERT-style models  
        elif hasattr(backbone, 'bert'):
            # Get embeddings from the embedding layer
            embeddings = backbone.bert.embeddings(
                input_ids=batch['input_ids'],
                token_type_ids=batch.get('token_type_ids')
            )
            
            # Use the full BERT model forward pass to get hidden states
            bert_outputs = backbone.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                token_type_ids=batch.get('token_type_ids'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = bert_outputs.last_hidden_state
            
        # For direct DistilBERT model instances (when backbone is the DistilBertModel itself)
        elif hasattr(backbone, 'embeddings') and hasattr(backbone, 'transformer') and 'distilbert' in str(type(backbone)).lower():
            # Direct DistilBERT architecture (e.g., when using base DistilBertModel)
            embeddings = backbone.embeddings(batch['input_ids'])
            
            # Get hidden states from the DistilBERT model
            distilbert_outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = distilbert_outputs.last_hidden_state

        # For DistilBERT-style models
        elif hasattr(backbone, 'distilbert'):
            # Get embeddings from the embedding layer
            embeddings = backbone.distilbert.embeddings(batch['input_ids'])
            
            # Use the full DistilBERT model forward pass to get hidden states
            distilbert_outputs = backbone.distilbert(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = distilbert_outputs.last_hidden_state
        
        # For LLAMA-style models (including Mistral)
        elif hasattr(backbone, 'model') and hasattr(backbone.model, 'embed_tokens'):
            # LLAMA/Mistral architecture
            embeddings = backbone.model.embed_tokens(batch['input_ids'])
            
            # Get hidden states from last layer before classification head
            outputs = backbone.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            
        # For direct LlamaModel instances (when backbone is the transformer model itself)
        elif hasattr(backbone, 'embed_tokens') and hasattr(backbone, 'layers'):
            # Direct LLAMA architecture (e.g., when using base LlamaModel)
            embeddings = backbone.embed_tokens(batch['input_ids'])
            
            # Get hidden states from the transformer model
            outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            
        # For GPT-style models (GPT-2, DialoGPT, etc.)
        elif hasattr(backbone, 'wte') and hasattr(backbone, 'h'):
            # GPT architecture - check for direct transformer model attributes
            embeddings = backbone.wte(batch['input_ids'])
            
            # Get hidden states from transformer
            outputs = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            
        else:
            raise ValueError(f"Unsupported model architecture: {type(backbone)}. "
                            f"Available attributes: {[attr for attr in dir(backbone) if not attr.startswith('_')]}. "
                            f"Please add support for this model type.")
        
        return embeddings, hidden_states

    def _extract_cls_representation(self, hidden_states: torch.Tensor, batch: Dict[str, torch.Tensor], model_backbone=None) -> torch.Tensor:
        """
        Extract CLS token representation from hidden states.
        
        Args:
            hidden_states: Hidden states tensor [batch_size, seq_len, hidden_dim]
            batch: Input batch dictionary
            model_backbone: The model backbone to determine architecture type
            
        Returns:
            CLS representations [batch_size, hidden_dim]
        """
        # Determine model type for appropriate representation extraction
        is_llama_style = False
        
        if model_backbone is not None:
            model_type_str = str(type(model_backbone)).lower()
            # Check for LLAMA-style models (including Mistral, Alpaca, etc.)
            is_llama_style = any(model_name in model_type_str for model_name in 
                               ['llama', 'mistral', 'alpaca', 'vicuna', 'codellama'])
            
            # Also check for direct LLAMA attributes
            if not is_llama_style:
                is_llama_style = (hasattr(model_backbone, 'embed_tokens') and hasattr(model_backbone, 'layers')) or \
                               (hasattr(model_backbone, 'model') and hasattr(model_backbone.model, 'embed_tokens'))
        
        if is_llama_style and 'attention_mask' in batch:
            # For LLAMA-style models, use mean pooling over all tokens
            # This is often more meaningful than just the last token for similarity analysis
            logger.debug(f"Using mean pooling extraction for LLAMA-style model: {type(model_backbone)}")
            
            attention_mask = batch['attention_mask']  # [batch_size, seq_len]
            
            # Mean pooling: average over all non-padding tokens
            # This captures the full sequence representation, not just the last token
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            sum_embeddings = torch.sum(hidden_states * expanded_mask, dim=1)
            sum_mask = torch.sum(expanded_mask, dim=1)
            cls_representations = sum_embeddings / (sum_mask + 1e-9)  # Add epsilon to avoid division by zero
            
            # Alternative last-token approach (commented out for now)
            # last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
            # last_token_indices = torch.clamp(last_token_indices, 0, hidden_states.size(1) - 1)
            # batch_size = hidden_states.size(0)
            # cls_representations = hidden_states[torch.arange(batch_size), last_token_indices]
            
        else:
            # For BERT/RoBERTa/DistilBERT-style models, use CLS token at position 0
            logger.debug(f"Using CLS-token (position 0) extraction for model: {type(model_backbone) if model_backbone else 'Unknown'}")
            cls_representations = hidden_states[:, 0, :]
            
        return cls_representations

    def _evaluate_model(
        self,
        model,
        dataloader,
        device,
        is_hans: bool = False,
        desc: str = "Evaluating",
        metrics: List[str] = None,
        save_predictions: bool = False,
        compute_confidence: bool = False,
        confidence_target_label: Optional[Union[int, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Core evaluation function moved from evaluate_utils.py
        """
        if metrics is None:
            metrics = ["accuracy"]
            
        metric_modules = {m: evaluate.load(m) for m in metrics}
        total_loss = 0
        model.eval()

        # Log model state
        logger.info(f"Model type: {type(model).__name__}")
        if hasattr(model, "active_adapters"):
            logger.info(f"Active adapters: {model.active_adapters}")
        if hasattr(model, "config"):
            logger.info(f"Model config type: {type(model.config).__name__}")

        all_predictions = []
        all_labels = []
        
        # Log a few predictions for the first batch
        first_batch_logged = False

        with torch.no_grad():
            model_device = self._get_model_device(model)
            for batch in tqdm(dataloader, desc=desc):
                # Always move to the model's actual device to avoid mismatches
                batch = self._move_batch_to_device(batch, model_device)
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                loss = outputs.loss
                
                # Log first batch predictions
                if not first_batch_logged:
                    logger.info("\nFirst batch details:")
                    logger.info(f"Logits shape: {outputs.logits.shape}")
                    logger.info(f"First few logits: {outputs.logits[:3]}")
                    logger.info(f"First few predictions: {predictions[:3]}")
                    logger.info(f"First few labels: {batch['labels'][:3]}")
                    first_batch_logged = True

                # Merge neutral and contradiction into one class (non-entailment)
                if is_hans:
                    non_entailment_logits = torch.max(
                        outputs.logits[:, 1], outputs.logits[:, 2]
                    ).unsqueeze(1)
                    entailment_logits = outputs.logits[:, 0].unsqueeze(1)
                    new_logits = torch.cat(
                        [entailment_logits, non_entailment_logits], dim=1
                    )        
                    predictions = torch.argmax(new_logits, dim=1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
                total_loss += loss.item()

        # Log prediction distribution
        unique_preds, pred_counts = torch.unique(torch.tensor(all_predictions), return_counts=True)
        logger.info("\nPrediction distribution:")
        for pred, count in zip(unique_preds.tolist(), pred_counts.tolist()):
            logger.info(f"Class {pred}: {count} predictions ({count/len(all_predictions)*100:.2f}%)")

        results = {}
        for name, metric in metric_modules.items():
            result = metric.compute(
                predictions=all_predictions, references=all_labels
            )
            if result is not None:
                results[name] = result[name]

        results["loss"] = total_loss / len(dataloader)
        
        # Compute confidence metrics if requested
        if compute_confidence and confidence_target_label is not None:
            logger.info("Computing confidence metrics for backdoor analysis...")
            confidence_results = self.compute_confidence_metrics(
                model=model,
                dataloader=dataloader,
                target_label=confidence_target_label,
                desc=f"{desc} - Confidence Analysis"
            )
            results["confidence_metrics"] = confidence_results
        
        # Only include predictions and labels if requested
        if save_predictions:
            results["labels"] = all_labels
            results["predictions"] = all_predictions

        return results 
    
    
    def validate_shallow_model(
        self,
        shallow_model,
        dataloader,
        num_bins: int = 10,
        desc: str = "Validating Shallow Model (f_b)"
    ) -> Dict[str, Any]:
        """
        Runs a special evaluation on the shallow model (f_b) to check its
        confidence distribution for the self-debiasing framework.
        
        This generates the data for a histogram like Figure 3 in the paper.
        
        Args:
            shallow_model: The f_b model to validate
            dataloader: The dataloader (e.g., full training set or a dev set)
            num_bins: Number of bins for the histogram
            
        Returns:
            A results dictionary containing histogram data and accuracy.
        """
        shallow_model.eval()
        model_device = self._get_model_device(shallow_model)
        
        all_probs = []
        all_labels = []

        logger.info(f"Running shallow model validation for {desc}...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = self._move_batch_to_device(batch, model_device)
                labels = batch["labels"]
                
                # Get logits
                # Create a copy of the batch without labels for the model
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                
                outputs = shallow_model(**model_inputs)
                logits = outputs.logits
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # --- Start Histogram Computation ---
        
        # Get confidence and predictions
        confidences, predictions = torch.max(all_probs, dim=-1)
        
        # Get a boolean mask of correct predictions
        correct_mask = (predictions == all_labels)
        
        # Get confidences for correct and incorrect predictions
        correct_confidences = confidences[correct_mask]
        
        # Define bins
        bin_range = (0.0, 1.0)
        
        import numpy as np
        
        counts_all, bin_edges = np.histogram(
            confidences.numpy(), bins=num_bins, range=bin_range
        )
        counts_correct, _ = np.histogram(
            correct_confidences.numpy(), bins=num_bins, range=bin_range
        )
        
        accuracy = correct_mask.float().mean().item()

        logger.info(f"Shallow Model Validation: Accuracy = {accuracy:.4f}")
        logger.info(f"Shallow Model Validation: Confidence histogram (all preds, 10 bins): {counts_all.tolist()}")
        logger.info(f"Shallow Model Validation: Confidence histogram (correct preds, 10 bins): {counts_correct.tolist()}")

        # This is the dictionary you can return and log to wandb
        total_preds = counts_all.sum()
        total_correct = counts_correct.sum()

        # Percentage of predictions in the highest-confidence (last) bin
        overconf = float(counts_all[-1] / total_preds * 100) if total_preds > 0 else 0.0
        # Same metric but restricted to correct predictions (useful to inspect calibration)
        overconf_correct = float(counts_correct[-1] / total_correct * 100) if total_correct > 0 else 0.0

        results = {
            "shallow_model_accuracy": accuracy,
            "shallow_model_confidence_histogram": {
            "all_predictions": {
                "counts": counts_all.tolist(),
                "bin_edges": bin_edges.tolist(),
                "overconf_percent": overconf
            },
            "correct_predictions": {
                "counts": counts_correct.tolist(),
                "bin_edges": bin_edges.tolist(),
                "overconf_percent": overconf_correct
            }
            }
        }
        
        return results