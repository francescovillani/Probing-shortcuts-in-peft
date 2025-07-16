import evaluate
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, List, Union, Optional, Any

logger = logging.getLogger(__name__)


def compute_embedding_similarities(
    model,
    clean_dataloader,
    triggered_dataloader,
    device,
    desc="Computing embedding similarities"
):
    """
    Compute cosine similarities between clean and triggered versions of examples.
    
    Args:
        model: The trained model
        clean_dataloader: DataLoader with clean examples
        triggered_dataloader: DataLoader with triggered examples (same order as clean)
        device: Device to run computations on
        desc: Description for progress bar
    
    Returns:
        Dictionary with similarity metrics at different embedding levels
    """
    model.eval()
    
    embedding_similarities = []  # After tokenization (embedding layer)
    hidden_similarities = []     # Before classification head (last hidden state)
    
    logger.info("Computing embedding similarities between clean and triggered examples...")
    
    with torch.no_grad():
        # Zip the dataloaders to process corresponding clean and triggered examples together
        paired_batches = zip(clean_dataloader, triggered_dataloader)
        
        for clean_batch, triggered_batch in tqdm(paired_batches, desc=desc):
            # Move batches to device
            clean_batch = {k: v.to(device) for k, v in clean_batch.items()}
            triggered_batch = {k: v.to(device) for k, v in triggered_batch.items()}
            
            # Extract embeddings and hidden states for clean examples
            clean_embeddings, clean_hidden = extract_embeddings_and_hidden_states(model, clean_batch)
            
            # Extract embeddings and hidden states for triggered examples  
            triggered_embeddings, triggered_hidden = extract_embeddings_and_hidden_states(model, triggered_batch)
            
            # Compute cosine similarities for embeddings (after tokenization)
            # Use mean of all token embeddings for sentence-level representation
            
            # Get attention masks to ignore padding tokens
            clean_mask = clean_batch['attention_mask'].unsqueeze(-1).expand(clean_embeddings.size()).float()
            triggered_mask = triggered_batch['attention_mask'].unsqueeze(-1).expand(triggered_embeddings.size()).float()

            # Zero out padding tokens
            clean_masked_embeddings = clean_embeddings * clean_mask
            triggered_masked_embeddings = triggered_embeddings * triggered_mask

            # Calculate sum and count of non-padding tokens
            clean_sum_embeddings = torch.sum(clean_masked_embeddings, dim=1)
            triggered_sum_embeddings = torch.sum(triggered_masked_embeddings, dim=1)
            
            # Clamp token count to avoid division by zero
            clean_token_count = clean_mask.sum(dim=1).clamp(min=1e-9)
            triggered_token_count = triggered_mask.sum(dim=1).clamp(min=1e-9)

            # Calculate mean embeddings
            clean_mean_embeddings = clean_sum_embeddings / clean_token_count
            triggered_mean_embeddings = triggered_sum_embeddings / triggered_token_count
            
            emb_similarities = F.cosine_similarity(clean_mean_embeddings, triggered_mean_embeddings, dim=1)
            embedding_similarities.extend(emb_similarities.cpu().tolist())
            
            # Compute cosine similarities for hidden states (before classification)
            # Use CLS token representation from last hidden state
            clean_hidden_cls = clean_hidden[:, 0, :]  # [batch_size, hidden_dim]
            triggered_hidden_cls = triggered_hidden[:, 0, :]  # [batch_size, hidden_dim]
            
            hidden_sims = F.cosine_similarity(clean_hidden_cls, triggered_hidden_cls, dim=1)
            hidden_similarities.extend(hidden_sims.cpu().tolist())
    
    # Compute statistics
    results = {
        # "embedding_similarities": {
        #     "mean": float(torch.tensor(embedding_similarities).mean()),
        #     "std": float(torch.tensor(embedding_similarities).std()),
        #     "min": float(torch.tensor(embedding_similarities).min()),
        #     "max": float(torch.tensor(embedding_similarities).max()),
        #     "count": len(embedding_similarities)
        # },
        "hidden_similarities": {
            "mean": float(torch.tensor(hidden_similarities).mean()),
            "std": float(torch.tensor(hidden_similarities).std()),
            "min": float(torch.tensor(hidden_similarities).min()),
            "max": float(torch.tensor(hidden_similarities).max()),
            "count": len(hidden_similarities)
        }
    }
    
    logger.info(f"Embedding similarities - Mean: {results['embedding_similarities']['mean']:.4f}, "
               f"Std: {results['embedding_similarities']['std']:.4f}")
    logger.info(f"Hidden similarities - Mean: {results['hidden_similarities']['mean']:.4f}, "
               f"Std: {results['hidden_similarities']['std']:.4f}")
    
    return results


def extract_embeddings_and_hidden_states(model, batch):
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
    
    # For RoBERTa-style models
    if hasattr(backbone, 'roberta'):
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
        
    else:
        raise ValueError(f"Unsupported model architecture: {type(backbone)}. "
                        f"Please add support for this model type.")
    
    return embeddings, hidden_states


def compute_confidence_metrics(
    model,
    dataloader,
    device,
    target_label: Union[int, List[int]],
    desc="Computing confidence metrics"
):
    """
    Compute confidence scores and logit differences for backdoor strength analysis.
    
    This provides continuous, lower-variance measurements of backdoor effectiveness
    compared to simple accuracy metrics.
    
    Args:
        model: The trained model
        dataloader: DataLoader with examples to analyze
        device: Device to run computations on
        target_label: Target label(s) for backdoor. Can be int or list of ints
        desc: Description for progress bar
    
    Returns:
        Dictionary with confidence metrics:
        - target_confidence: Statistics about confidence in target label
        - logit_differences: Statistics about target-true logit differences
        - prediction_distribution: How predictions are distributed
    """
    model.eval()
    
    target_confidences = []      # Softmax probability for target label
    logit_differences = []       # Difference between target and true label logits
    target_predictions = 0       # Count of predictions matching target
    total_samples = 0
    
    # Handle both single target and multi-target cases
    if isinstance(target_label, int):
        target_labels = [target_label]
    else:
        target_labels = target_label
    
    logger.info(f"Computing confidence metrics for target label(s): {target_labels}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            true_labels = batch["labels"]
            
            # Compute softmax probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Process each sample in the batch
            for i in range(logits.size(0)):
                sample_logits = logits[i]
                sample_probs = probabilities[i]
                true_label = true_labels[i].item()
                
                # For multi-target backdoors, determine the appropriate target
                # If original sample had label in target_labels, that's the target
                if true_label in target_labels:
                    current_target = true_label
                else:
                    # For cross-label attacks, use the first target label
                    current_target = target_labels[0]
                
                # Target confidence (softmax probability for target label)
                target_conf = sample_probs[current_target].item()
                target_confidences.append(target_conf)
                
                # Logit difference (target logit - true label logit)
                if current_target != true_label:
                    logit_diff = sample_logits[current_target].item() - sample_logits[true_label].item()
                    logit_differences.append(logit_diff)
                else:
                    # If target equals true label, difference is 0
                    logit_differences.append(0.0)
                
                # Track if model predicts target
                predicted = torch.argmax(sample_logits).item()
                if predicted == current_target:
                    target_predictions += 1
                
                total_samples += 1
    
    # Compute comprehensive statistics
    target_conf_tensor = torch.tensor(target_confidences)
    logit_diff_tensor = torch.tensor(logit_differences)
    
    results = {
        "target_confidence": {
            "mean": float(target_conf_tensor.mean()),
            "std": float(target_conf_tensor.std()),
            "min": float(target_conf_tensor.min()),
            "max": float(target_conf_tensor.max()),
            "median": float(target_conf_tensor.median()),
            "q25": float(target_conf_tensor.quantile(0.25)),
            "q75": float(target_conf_tensor.quantile(0.75)),
            "count": len(target_confidences)
        },
        "logit_differences": {
            "mean": float(logit_diff_tensor.mean()),
            "std": float(logit_diff_tensor.std()),
            "min": float(logit_diff_tensor.min()),
            "max": float(logit_diff_tensor.max()),
            "median": float(logit_diff_tensor.median()),
            "q25": float(logit_diff_tensor.quantile(0.25)),
            "q75": float(logit_diff_tensor.quantile(0.75)),
            "count": len(logit_differences)
        },
        "prediction_stats": {
            "target_prediction_rate": target_predictions / total_samples if total_samples > 0 else 0.0,
            "total_samples": total_samples,
            "target_predictions": target_predictions
        }
    }
    
    # Log summary
    logger.info(f"Confidence Metrics Summary:")
    logger.info(f"  Target Confidence - Mean: {results['target_confidence']['mean']:.4f}, "
               f"Std: {results['target_confidence']['std']:.4f}")
    logger.info(f"  Logit Differences - Mean: {results['logit_differences']['mean']:.4f}, "
               f"Std: {results['logit_differences']['std']:.4f}")
    logger.info(f"  Target Prediction Rate: {results['prediction_stats']['target_prediction_rate']:.4f}")
    
    return results


def evaluate_model(
    model,
    dataloader,
    device,
    is_hans = False,
    desc="Evaluating",
    metrics=["accuracy", "f1", "precision", "recall"],
    save_predictions=False,
    compute_confidence=False,
    confidence_target_label: Optional[Union[int, List[int]]] = None,
):
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
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
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
        if name in {"f1", "precision", "recall"}:
            results[name] = metric.compute(
                predictions=all_predictions, references=all_labels, average="macro"
            )[name]
        else:
            results[name] = metric.compute(
                predictions=all_predictions, references=all_labels
            )[name]

    results["loss"] = total_loss / len(dataloader)
    
    # Compute confidence metrics if requested
    if compute_confidence and confidence_target_label is not None:
        logger.info("Computing confidence metrics for backdoor analysis...")
        confidence_results = compute_confidence_metrics(
            model=model,
            dataloader=dataloader,
            device=device,
            target_label=confidence_target_label,
            desc=f"{desc} - Confidence Analysis"
        )
        results["confidence_metrics"] = confidence_results
    
    # Only include predictions and labels if requested
    if save_predictions:
        results["labels"] = all_labels
        results["predictions"] = all_predictions

    return results
