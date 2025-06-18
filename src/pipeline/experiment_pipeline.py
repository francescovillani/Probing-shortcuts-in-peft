from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import yaml
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from data.dataset import DatasetManager
from data.dataset_modifiers import inject_trigger_into_dataset
from config.config_schema import (
    TrainingConfig,
    DatasetConfig,
    ModelConfig,
    PEFTConfig,
    WandBConfig,
    load_and_validate_config,
)
from models.peft_factory import get_peft_model_factory
from train_and_eval import TrainingRunner


logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    # Dataset Generation
    base_dataset: DatasetConfig
    poisoning: Dict[str, Dict[str, Any]]  # Different poisoning configs by name
    save_datasets: bool
    dataset_save_dir: Optional[str]
    
    # Model & Training
    model_config: ModelConfig
    training_config: Dict[str, Any]
    max_length: int
    
    # Evaluation
    evaluation_datasets: Dict[str, DatasetConfig]
    
    # Logging
    wandb_config: WandBConfig = field(default_factory=WandBConfig)
    output_dir: str = "outputs/experiments"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from a YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to appropriate config objects
        base_dataset = DatasetConfig(**config_dict['base_dataset'])
        model_config = ModelConfig(
            base_model=config_dict['model']['base_model'],
            peft_config=PEFTConfig(**config_dict['model']['peft'])
        )
        wandb_config = WandBConfig(**config_dict['wandb'])
        
        # Convert evaluation dataset configs
        eval_datasets = {}
        for name, eval_config in config_dict['evaluation_datasets'].items():
            eval_datasets[name] = DatasetConfig(**eval_config)
        
        return cls(
            base_dataset=base_dataset,
            poisoning=config_dict.get('poisoning', {}),
            save_datasets=config_dict.get('save_datasets', False),
            dataset_save_dir=config_dict.get('dataset_save_dir'),
            model_config=model_config,
            training_config=config_dict['training'],
            max_length=config_dict['training'].get('max_length', 512),
            evaluation_datasets=eval_datasets,
            wandb_config=wandb_config,
            output_dir=config_dict.get('output_dir', 'outputs/experiments'),
            seed=config_dict.get('seed', 42)
        )

class ExperimentPipeline:
    def __init__(self, config: Union[PipelineConfig, str]):
        # Allow initialization with either config object or path to yaml
        self.config = (
            PipelineConfig.from_yaml(config) if isinstance(config, str)
            else config
        )
        self.datasets: Dict[str, Dataset] = {}
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s"
        )
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if self.config.save_datasets and self.config.dataset_save_dir:
            Path(self.config.dataset_save_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_wandb(self, run_name: Optional[str] = None):
        """Initialize W&B logging"""
        if self.config.wandb_config.enabled:
            wandb.init(
                project=self.config.wandb_config.project,
                name=run_name,
                config={
                    "model_config": self.config.model_config.model_dump(),
                    "training_config": self.config.training_config,
                    "poisoning_config": self.config.poisoning,
                }
            )
    
    def _apply_poisoning(self, dataset: Dataset, poison_config: Dict[str, Any], name: str) -> Dataset:
        """Apply poisoning configuration to a dataset"""
        logger.info(f"Applying {name} poisoning configuration...")
        poisoned_dataset = inject_trigger_into_dataset(
            dataset=dataset,
            **poison_config
        )
        
        # Save poisoned dataset if requested
        if self.config.save_datasets:
            save_path = Path(self.config.dataset_save_dir) / f"poisoned_{name}"
            poisoned_dataset.save_to_disk(str(save_path))
            logger.info(f"Saved poisoned dataset to {save_path}")
        
        return poisoned_dataset
    
    def generate_datasets(self) -> Dict[str, Dataset]:
        """Generate datasets including poisoned versions if specified"""
        logger.info("Generating datasets...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_config.base_model
        )
        
        # Initialize dataset manager
        dataset_manager = DatasetManager(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            seed=self.config.seed
        )
        
        # Load and poison training dataset
        train_dataset = dataset_manager._load_dataset(self.config.base_dataset.model_dump())
        if 'train' in self.config.poisoning:
            train_dataset = self._apply_poisoning(
                train_dataset,
                self.config.poisoning['train'],
                'train'
            )
        self.datasets["train"] = train_dataset
        
        # Load and potentially poison evaluation datasets
        for name, eval_config in self.config.evaluation_datasets.items():
            eval_dataset = dataset_manager._load_dataset(eval_config.model_dump())
            
            # Apply poisoning if specified
            if hasattr(eval_config, 'poisoning') and eval_config.poisoning:
                poison_config = self.config.poisoning.get(eval_config.poisoning)
                if poison_config:
                    eval_dataset = self._apply_poisoning(
                        eval_dataset,
                        poison_config,
                        f"eval_{name}"
                    )
            
            self.datasets[f"eval_{name}"] = eval_dataset
        
        return self.datasets
    
    def setup_model(self):
        """Initialize model with PEFT if specified"""
        logger.info("Setting up model...")
        
        # Model
        self.logger.info(f"Using PEFT type: {self.config.model.peft_config.peft_type}")
        factory = get_peft_model_factory(
            peft_type=self.config.model.peft_config.peft_type,
            model_name=self.config.model.base_model,
            num_labels=self.config.num_labels,
            peft_args=self.config.model.peft_config.peft_args,
            device=self.device,
        )
        self.model = factory.create_model()
    
    def train_and_eval(self, logger=None):
        """Train and evaluate model using TrainingRunner from train_and_eval.py"""
        # Build a TrainingConfig dict compatible with TrainingRunner
        # Map pipeline config fields to TrainingConfig
        training_config_dict = {
            'model': self.config.model_config,
            'num_labels': self.config.training_config['num_labels'],
            'epochs': self.config.training_config['epochs'],
            'lr': self.config.training_config['lr'],
            'seed': self.config.seed,
            'outputdir': self.config.output_dir,
            'train_dataset': self.config.base_dataset,
            'validation_datasets': self.config.evaluation_datasets,
            'max_train_size': self.config.training_config.get('max_train_size', None),
            'tokenizer_max_length': self.config.training_config.get('max_length', 512),
            'gradient_accumulation_steps': self.config.training_config.get('gradient_accumulation_steps', 1),
            'warmup_ratio': self.config.training_config.get('warmup_ratio', 0.06),
            'save_strategy': self.config.training_config.get('save_strategy', 'epoch'),
            'metric_for_best_model': self.config.training_config.get('metric_for_best_model', 'accuracy'),
            'wandb': self.config.wandb_config,
        }
        # Validate config
        training_config = TrainingConfig(**{k: v for k, v in training_config_dict.items() if v is not None})
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        # Run training
        runner = TrainingRunner(training_config, self.config.output_dir, logger)
        results = runner.run_training()
        return results
    
    def run(self, experiment_name: Optional[str] = None):
        """Run full pipeline"""
        # Initialize W&B
        self._init_wandb(experiment_name)
        try:
            # Generate datasets (and poison if needed)
            self.generate_datasets()
            # Train and evaluate using TrainingRunner
            results = self.train_and_eval()
            # Log final results
            if self.config.wandb_config.enabled:
                wandb.log({"results": results})
            return {"results": results}
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            if self.config.wandb_config.enabled:
                wandb.finish() 