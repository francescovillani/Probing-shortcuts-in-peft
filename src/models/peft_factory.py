import torch
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    # RandLoraConfig,
    PeftModel,
    PeftConfig
)
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def log_trainable_parameters(model, method_name: str = "PEFT") -> None:
    """
    Log detailed information about trainable parameters in the model.
    
    Args:
        model: The model to analyze
        method_name: Name of the PEFT method for logging context
    """
    trainable_params = []
    frozen_params = []
    total_trainable = 0
    total_frozen = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_info = {
            'name': name,
            'shape': list(param.shape),
            'count': param_count,
            'dtype': str(param.dtype)
        }
        
        if param.requires_grad:
            trainable_params.append(param_info)
            total_trainable += param_count
        else:
            frozen_params.append(param_info)
            total_frozen += param_count
    
    total_params = total_trainable + total_frozen
    percentage = (total_trainable / total_params * 100) if total_params > 0 else 0
    
    logger.debug(f"\n=== {method_name} Trainable Parameters Analysis ===")
    logger.debug(f"Total parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {total_trainable:,} ({percentage:.2f}%)")
    logger.debug(f"Frozen parameters: {total_frozen:,}")
    
    if trainable_params:
        logger.debug(f"\nTrainable parameters ({len(trainable_params)} layers):")
        for param_info in trainable_params:
            logger.debug(f"  {param_info['name']}: {param_info['shape']} ({param_info['count']:,} params, {param_info['dtype']})")
    else:
        logger.debug("No trainable parameters found!")
    
    # Log a summary of frozen parameter types if there are many
    if len(frozen_params) > 20:
        logger.debug(f"\nFrozen parameters: {len(frozen_params)} layers (showing summary)")
        # Group by parameter type
        frozen_summary = {}
        for param_info in frozen_params:
            param_type = param_info['name'].split('.')[-1]  # Get the last part (weight, bias, etc.)
            if param_type not in frozen_summary:
                frozen_summary[param_type] = {'count': 0, 'total_params': 0}
            frozen_summary[param_type]['count'] += 1
            frozen_summary[param_type]['total_params'] += param_info['count']
        
        for param_type, summary in frozen_summary.items():
            logger.debug(f"  {param_type}: {summary['count']} layers, {summary['total_params']:,} params")
    elif frozen_params:
        logger.debug(f"\nFrozen parameters ({len(frozen_params)} layers):")
        for param_info in frozen_params[:10]:  # Show first 10 to avoid spam
            logger.debug(f"  {param_info['name']}: {param_info['shape']} ({param_info['count']:,} params)")
        if len(frozen_params) > 10:
            logger.debug(f"  ... and {len(frozen_params) - 10} more frozen layers")
    
    logger.debug("=" * 50)

def unfreeze_classification_head(base_model, num_labels, peft_args, method_name):
    """
    Unfreeze the classification head for prompt-based PEFT methods.
    
    This is critical because the classification head is randomly initialized when using
    AutoModelForSequenceClassification and needs to be trained.
    
    Args:
        base_model: The base model with the classification head
        num_labels: Expected number of output labels
        peft_args: PEFT arguments that may contain unfrozen_params or heads_to_save
        method_name: Name of the PEFT method for logging
        
    Returns:
        Tuple of (classification_heads_found, custom_unfrozen_count)
    """
    classification_heads_found = []
    
    # Allow user to specify heads to unfreeze
    user_specified_heads = peft_args.get("heads_to_save", [])
    
    if user_specified_heads:
        logger.info(f"[{method_name}] User specified heads to unfreeze: {user_specified_heads}")
        for head_name in user_specified_heads:
            if hasattr(base_model, head_name):
                head_module = getattr(base_model, head_name)
                param_count = sum(p.numel() for p in head_module.parameters())
                
                for param in head_module.parameters():
                    param.requires_grad = True
                
                classification_heads_found.append(head_name)
                logger.info(f"[{method_name}] Unfroze specified head '{head_name}' with {param_count:,} parameters")
            else:
                logger.warning(f"[{method_name}] Specified head '{head_name}' not found in the model.")
    else:
        # Determine model family and expected head names
        model_type = type(base_model).__name__
        if 'bert' in model_type.lower():
            expected_heads = ['classifier']
        elif 'roberta' in model_type.lower():
            expected_heads = ['classifier']
        elif 'electra' in model_type.lower():
            expected_heads = ['classifier']
        elif 'deberta' in model_type.lower():
            expected_heads = ['classifier']
        elif 'distilbert' in model_type.lower():
            expected_heads = ['classifier']
        elif any(arch in model_type.lower() for arch in ['gpt', 'opt', 'bloom', 'llama', 'mistral']):
            expected_heads = ['score']
        else:
            expected_heads = ['classifier', 'score', 'head']
        
        # Find and unfreeze classification heads
        for head_name in expected_heads:
            if hasattr(base_model, head_name):
                head_module = getattr(base_model, head_name)
                param_count = sum(p.numel() for p in head_module.parameters())
                
                for param in head_module.parameters():
                    param.requires_grad = True
                
                classification_heads_found.append(head_name)
                logger.info(f"[{method_name}] Unfroze classification head '{head_name}' with {param_count:,} parameters")
                break  # Only unfreeze the first match
        
        # Fallback: look for linear layers with correct output size if no standard head found
        if not classification_heads_found:
            logger.warning(f"[{method_name}] No standard classification heads found. Searching for linear layers...")
            for name, module in base_model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == num_labels:
                    param_count = sum(p.numel() for p in module.parameters())
                    for param in module.parameters():
                        param.requires_grad = True
                    classification_heads_found.append(name)
                    logger.info(f"[{method_name}] Unfroze linear layer '{name}' with {param_count:,} parameters")
                    break
    
    # Allow additional custom parameters to be unfrozen if specified
    unfrozen_params = peft_args.get("unfrozen_params", [])
    custom_unfrozen_count = 0
    
    if unfrozen_params:
        logger.info(f"[{method_name}] Attempting to unfreeze {len(unfrozen_params)} custom parameter path(s)")
        for param_path in unfrozen_params:
            if set_requires_grad_by_path(base_model, param_path):
                custom_unfrozen_count += 1
    
    return classification_heads_found, custom_unfrozen_count

# Base factory class
class PEFTModelFactory:
    def __init__(self, model_name, num_labels, peft_args=None, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.peft_args = peft_args or {}
        self.device = device

    def create_model(self):
        # Default: just load the base model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        if self.device:
            model = model.to(self.device)
        return model

# LoRA factory
class LoraModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        config = LoraConfig(**self.peft_args)
        model = get_peft_model(base_model, config)
        if self.device:
            model = model.to(self.device)
        return model

# QLoRA factory
class QLoraModelFactory(PEFTModelFactory):
    def _get_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["classifier", "pre_classifier"],
        )

    def create_model(self):
        quantization_config = self._get_quantization_config()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, quantization_config=quantization_config
        )
        base_model = prepare_model_for_kbit_training(base_model)
        config = LoraConfig(**self.peft_args)
        model = get_peft_model(base_model, config)
        if self.device:
            model = model.to(self.device)
        return model

# IA3 factory
class IA3ModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        config = IA3Config(**self.peft_args)
        model = get_peft_model(base_model, config)
        if self.device:
            model = model.to(self.device)
        return model

# Prompt Tuning factory
class PromptTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        # Freeze all parameters first
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        peft_config_args.pop("heads_to_save", None)
        peft_config_args.pop("unfrozen_params", None)
        config = PromptTuningConfig(**peft_config_args)
        model = get_peft_model(base_model, config)
        
        # Unfreeze the classification head after applying PEFT
        classification_heads_found, custom_unfrozen_count = unfreeze_classification_head(base_model, self.num_labels, self.peft_args, "Prompt Tuning")
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
        
        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"Prompt Tuning setup complete:")
        logger.info(f"  - Classification heads unfrozen: {classification_heads_found}")
        logger.info(f"  - Custom parameters unfrozen: {custom_unfrozen_count}")
        logger.info(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")
        
        if self.device:
            model = model.to(self.device)
        return model

# Prefix Tuning factory
class PrefixTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        # Freeze all parameters first
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        peft_config_args.pop("heads_to_save", None)
        peft_config_args.pop("unfrozen_params", None)
        config = PrefixTuningConfig(**peft_config_args)
        model = get_peft_model(base_model, config)
        
        # Unfreeze the classification head after applying PEFT
        classification_heads_found, custom_unfrozen_count = unfreeze_classification_head(base_model, self.num_labels, self.peft_args, "Prefix Tuning")
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
        
        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"Prefix Tuning setup complete:")
        logger.info(f"  - Classification heads unfrozen: {classification_heads_found}")
        logger.info(f"  - Custom parameters unfrozen: {custom_unfrozen_count}")
        logger.info(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")
        
        if self.device:
            model = model.to(self.device)
        return model

# P-Tuning v2 factory
class PTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        # Freeze all parameters first
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        peft_config_args.pop("heads_to_save", None)
        peft_config_args.pop("unfrozen_params", None)
        config = PromptEncoderConfig(**peft_config_args)
        model = get_peft_model(base_model, config)
        
        # Unfreeze the classification head after applying PEFT
        classification_heads_found, custom_unfrozen_count = unfreeze_classification_head(base_model, self.num_labels, self.peft_args, "P-Tuning v2")
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
        
        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(f"P-Tuning v2 setup complete:")
        logger.info(f"  - Classification heads unfrozen: {classification_heads_found}")
        logger.info(f"  - Custom parameters unfrozen: {custom_unfrozen_count}")
        logger.info(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")
        
        if self.device:
            model = model.to(self.device)
        return model

def set_requires_grad_by_path(model, param_path: str) -> bool:
    """
    Set requires_grad=True for a parameter at the given path.
    
    Args:
        model: The model to modify
        param_path: Dot-separated path to the parameter (e.g., 'classifier.dense.weight')
    
    Returns:
        bool: True if parameter was found and modified, False otherwise
    """
    parts = param_path.split('.')
    obj = model
    try:
        for part in parts:
            obj = getattr(obj, part)
        obj.requires_grad = True
        logger.info(f"Successfully set requires_grad=True for parameter: {param_path}")
        return True
    except AttributeError as e:
        logger.warning(f"Could not set requires_grad for {param_path}: {e}")
        return False

    """
    Get detailed information about the model architecture for debugging.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with architecture information
    """
    info = {
        'model_type': type(model).__name__,
        'config_type': type(model.config).__name__ if hasattr(model, 'config') else 'Unknown',
        'available_attributes': [attr for attr in dir(model) if not attr.startswith('_')],
        'linear_layers': [],
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }
    
    # Collect information about linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            info['linear_layers'].append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'has_bias': module.bias is not None
            })
    
    return info

# BitFit factory
class BitFitModelFactory(PEFTModelFactory):
    def create_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze all bias terms in linear layers  
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.requires_grad = True
                
        classification_heads_found, custom_unfrozen_count = unfreeze_classification_head(model, self.num_labels, self.peft_args, "BitFit")
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found

        unfrozen_params = self.peft_args.get("unfrozen_params", [])
        for param_path in unfrozen_params:
            set_requires_grad_by_path(model, param_path)
                
        if self.device:
            model = model.to(self.device)
        return model

# Classifier-only factory
class ClassifierOnlyModelFactory(PEFTModelFactory):
    def create_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        
        # Get model architecture information for logging
        model_type = type(model).__name__
        config_type = type(model.config).__name__ if hasattr(model, 'config') else 'Unknown'
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Model architecture: {model_type} ({config_type})")
        logger.debug(f"Model has {total_params:,} total parameters")
        
        # Since we use AutoModelForSequenceClassification, we're guaranteed to have a proper classification head
        # The head will be named according to the model family's convention
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        classification_heads_found, custom_unfrozen_count = unfreeze_classification_head(model, self.num_labels, self.peft_args, "Classifier-only")
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
        
        if self.device:
            model = model.to(self.device)
        return model

# Load PEFT model factory
class LoadPeftModelFactory(PEFTModelFactory):
    def create_model(self):
        peft_model_path = self.peft_args.get("peft_model_path")
        if not peft_model_path:
            raise ValueError("peft_model_path must be provided to load a model")
            
        # Try to detect if this is a PEFT model by checking for adapter_config.json
        is_peft_model = (Path(peft_model_path) / "adapter_config.json").exists()
        
        if is_peft_model:
            # Load PEFT config first to get base model configuration
            config = PeftConfig.from_pretrained(peft_model_path)
            # For other PEFT methods, use standard loading
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                num_labels=self.num_labels,
                torch_dtype=config.torch_dtype if hasattr(config, 'torch_dtype') else torch.float32
            )
            model = PeftModel.from_pretrained(base_model, peft_model_path, is_trainable=False)
        else:
            # Load as regular fine-tuned model
            model = AutoModelForSequenceClassification.from_pretrained(
                peft_model_path, num_labels=self.num_labels
            )
            
        if self.device:
            model = model.to(self.device)
        
        # Load classifier head if it exists
        custom_config_path = Path(peft_model_path) / "custom_config.json"
        
        if custom_config_path.exists():
            import json
            with open(custom_config_path, 'r') as f:
                custom_config = json.load(f)
            
            head_names = custom_config.get("classification_heads_to_save", [])
            if head_names:
                model.classification_heads_to_save = head_names
                # The base model holds the classification head
                base_model = model.base_model if hasattr(model, 'base_model') else model
                
                for head_name in head_names:
                    classifier_weights_path = Path(peft_model_path) / f"{head_name}.pt"
                    if not classifier_weights_path.exists():
                        logger.warning(f"Could not find weights file for classification head '{head_name}' at {classifier_weights_path}")
                        continue
                        
                    if hasattr(base_model, head_name):
                        head_module = getattr(base_model, head_name)
                        logger.info(f"Loading custom classification head '{head_name}' from {classifier_weights_path}")
                        state_dict = torch.load(classifier_weights_path, map_location=self.device)
                        head_module.load_state_dict(state_dict)
                    else:
                        logger.warning(f"Could not find classification head '{head_name}' in the model.")

        return model


# Registry
PEFT_FACTORIES = {
    "none": PEFTModelFactory,
    "lora": LoraModelFactory,
    "qlora": QLoraModelFactory,
    "ia3": IA3ModelFactory,
    "prompt_tuning": PromptTuningModelFactory,
    "prefix_tuning": PrefixTuningModelFactory,
    "p_tuning": PTuningModelFactory,
    "bitfit": BitFitModelFactory,
    "classifier_only": ClassifierOnlyModelFactory,
    "load_peft": LoadPeftModelFactory,
}

def get_peft_model_factory(peft_type, model_name, num_labels, peft_args=None, device=None):

    factory_cls = PEFT_FACTORIES.get(peft_type, PEFTModelFactory)
    factory = factory_cls(model_name, num_labels, peft_args, device)
    # Store original peft_type for loading 
    factory.original_peft_type = peft_type
    
    # Override the create_model method to add debug logging
    original_create_model = factory.create_model
    
    def create_model_with_debug():
        model = original_create_model()
        # Log trainable parameters in debug mode
        log_trainable_parameters(model, f"{peft_type.upper()}")
        return model
    
    factory.create_model = create_model_with_debug
    return factory
