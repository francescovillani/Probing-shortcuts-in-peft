import evaluate
import torch
from tqdm import tqdm


def evaluate_model(
    model,
    dataloader,
    device,
    is_hans,
    desc="Evaluating",
    metrics=["accuracy", "f1", "precision", "recall"],
):
    metric_modules = {m: evaluate.load(m) for m in metrics}
    total_loss = 0
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            loss = outputs.loss
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
    results["labels"] = all_labels
    results["predictions"] = all_predictions

    return results
