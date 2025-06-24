# Probing Shortcuts in PEFT

A research framework for studying shortcut learning in Parameter-Efficient Fine-Tuning (PEFT) techniques, built on Hugging Face and Weights & Biases.

## Core Features

- **YAML-First Configuration**: Define entire experiments in simple, reusable YAML files.
- **Unified CLI**: A single, clean command-line interface for all operations.
- **Service-Oriented**: Modular and extensible architecture for managing datasets, models, and experiments.
- **Dataset Poisoning**: Built-in tools for injecting triggers and studying shortcut learning.
- **WandB Integration**: Seamless support for experiment tracking and hyperparameter sweeps.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Login to Weights & Biases (for logging and sweeps)
wandb login
```

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