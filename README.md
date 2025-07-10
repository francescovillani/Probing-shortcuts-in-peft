# Probing Shortcuts in PEFT

A research framework for studying shortcut learning in Parameter-Efficient Fine-Tuning (PEFT) techniques, built on Hugging Face and Weights & Biases.

## Core Features

- **YAML-First Configuration**: Define entire experiments in simple, reusable YAML files.
- **Unified CLI**: A single, clean command-line interface for all operations.
- **Service-Oriented**: Modular and extensible architecture for managing datasets, models, and experiments.
- **Dataset Poisoning**: Built-in tools for injecting triggers and studying shortcut learning.
- **WandB Integration**: Seamless support for experiment tracking and hyperparameter sweeps.
- **Debug Samples**: Automatic extraction of representative dataset samples for verification and comparison.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/francescovillani/Probing-shortcuts-in-peft.git
cd Probing-shortcuts-in-peft

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to Weights & Biases (for logging and sweeps)
wandb login
```

## Debug Samples Feature

The framework automatically extracts a small set of representative text samples from each dataset for debugging and verification purposes. This feature helps you:

- **Verify Data Processing**: Check that poisoning/triggers are applied correctly
- **Compare Experiments**: Ensure different runs use the same data splits by comparing samples
- **Debug Issues**: Quickly inspect actual text content and labels from your datasets

### Configuration

```yaml
# Enable debug sample extraction (default: true)
extract_debug_samples: true

# Number of samples to extract per dataset (default: 5, range: 1-20)
num_debug_samples: 5
```

### What's Captured

For each dataset (training, validation, etc.), the framework deterministically samples examples and captures:

- **Raw text content** (before tokenization)
- **Labels** (both original and processed)
- **Trigger status** (whether the sample contains injected triggers)
- **Dataset metadata** (column names, indices)

### Example Output

```json
{
  "debug_samples": {
    "training": [
      {
        "index": 42,
        "text_content": {
          "sentence": "cf debug this movie is excellent and engaging"
        },
        "label": 1,
        "has_trigger": true,
        "raw_example_keys": ["sentence", "label", "idx", "has_trigger"]
      }
    ],
    "clean_test": [...],
    "poisoned_test": [...]
  }
}
```

### Benefits

1. **Deterministic Sampling**: Same seed + same dataset = same samples across runs
2. **Poisoning Verification**: Easily see which samples have triggers and verify correct injection
3. **Cross-Run Comparison**: Compare debug samples between experiments to ensure data consistency
4. **Quick Debugging**: Inspect actual text without diving into dataset internals

## Embedding Similarity Analysis

The framework includes a powerful feature for analyzing how triggers affect model representations at different levels. This helps you understand where and how much triggers impact your model's internal embeddings.

### What It Computes

- **Embedding Similarities**: Cosine similarity between clean and triggered versions at the embedding layer (after tokenization)
- **Hidden Similarities**: Cosine similarity between clean and triggered versions at the hidden state level (before classification head)
- **Statistics**: Mean, standard deviation, min, max, and count for both similarity types

### Configuration

```yaml
# Enable embedding similarity analysis during training (default: false)
compute_embedding_similarities: true
```

### Requirements

- At least one validation dataset must have `poisoning.enabled: true`
- The poisoned dataset should contain triggered examples (`injection_percentage > 0`)

### Example Usage

```yaml
validation_datasets:
  shortcut_validation:
    name: "nyu-mll/glue"
    config: "sst2"
    split: "validation"
    batch_size: 32
    is_hf: true
    poisoning:
      enabled: true
      text_column_names: ["sentence"]
      trigger_tokens: ["xp"]
      injection_percentage: 1.0  # Poison all samples for analysis
      injection_position: "start"
      target_label: 0
      filter_labels: [0]

# Enable similarity computation
compute_embedding_similarities: true
```

### Interpreting Results

```json
{
  "embedding_similarities": {
    "mean": 0.923,    // Higher values (closer to 1.0) = minimal trigger impact
    "std": 0.034,     // Lower std = consistent impact across examples
    "min": 0.812,
    "max": 0.987,
    "count": 1000
  },
  "hidden_similarities": {
    "mean": 0.887,    // Compare with embedding similarities
    "std": 0.052,     // to see where triggers have most impact
    "min": 0.723,
    "max": 0.975,
    "count": 1000
  }
}
```

**Analysis Tips**:
- **High similarity (>0.9)**: Triggers have minimal impact on representations
- **Low similarity (<0.7)**: Triggers significantly alter representations  
- **Embedding vs Hidden comparison**: Shows whether triggers affect early (embedding) or late (hidden) representations more
- **Standard deviation**: Indicates consistency of trigger effects across examples

### Where to Find Results

Results are automatically saved in your experiment output directory:
- `results/experiment_results.json` - Contains full similarity statistics
- WandB dashboard - Real-time similarity metrics during training
- Terminal logs - Summary statistics for each epoch

## Core Commands

All commands are run through the unified CLI script `src/cli.py`. See the `examples/configs/` directory for detailed and documented configuration files.

### 1. Train a Model

Use the `train` command to start an experiment. The configuration is defined in a YAML file.

```bash
python src/cli.py train --config examples/configs/01_lora_finetune.yml
```

> See `examples/configs/01_lora_finetune.yml` for a basic training setup and `02_poisoning_experiment.yml` for an example with dataset poisoning.

### 2. Evaluate a Trained Model

Use the `evaluate` command to test saved model checkpoints.

```bash
python src/cli.py evaluate --config examples/configs/03_evaluation.yml
```

> **Note**: You must update the `checkpoints_dir` path inside `03_evaluation.yml` to point to the directory where your trained model was saved.

### 3. Run a Hyperparameter Sweep

This framework uses a two-step process to run `wandb` sweeps:

**Step 1: Create the sweep.**  
This requires a *base config* (like `01_lora_finetune.yml`) and a *sweep config* that defines the parameter search space.

```bash
python src/cli.py sweep \
  --config examples/configs/01_lora_finetune.yml \
  --sweep-config examples/configs/04_wandb_sweep.yml
```

**Step 2: Run the `wandb` agent.**  
Copy the sweep ID printed by the previous command and use it to start one or more agents.

```bash
# Replace <SWEEP_ID> with the ID from Step 1
wandb agent <SWEEP_ID>
```

The agent will now execute training runs with different hyperparameter combinations, with all results tracked in your WandB dashboard.