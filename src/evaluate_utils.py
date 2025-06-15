import evaluate
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate_model(
    model,
    dataloader,
    device,
    is_hans = False,
    desc="Evaluating",
    metrics=["accuracy", "f1", "precision", "recall"],
    save_predictions=False,
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
    
    # Only include predictions and labels if requested
    if save_predictions:
        results["labels"] = all_labels
        results["predictions"] = all_predictions

    return results
