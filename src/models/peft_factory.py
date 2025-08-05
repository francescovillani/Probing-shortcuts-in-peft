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
        config = PromptTuningConfig(**self.peft_args)
        model = get_peft_model(base_model, config)
        if self.device:
            model = model.to(self.device)
        return model

# Prefix Tuning factory
class PrefixTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        config = PrefixTuningConfig(**self.peft_args)
        model = get_peft_model(base_model, config)
        if self.device:
            model = model.to(self.device)
        return model

# P-Tuning v2 factory
class PTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        config = PromptEncoderConfig(**self.peft_args)
        model = get_peft_model(base_model, config)
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

def find_classification_head_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    """
    Find all potential classification head modules in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        List of (name, module) tuples for potential classification heads
    """
    classification_heads = []
    
    # Common classification head names across different architectures
    common_head_names = [
        'classifier', 'score', 'cls', 'classification_head', 'head',
        'lm_head', 'qa_outputs', 'seq_relationship', 'pooler_output'
    ]
    
    # Architecture-specific patterns
    model_type = type(model).__name__.lower()
    
    # Add model-specific head names
    if 'bert' in model_type:
        common_head_names.extend(['pooler', 'pre_classifier'])
    elif 'roberta' in model_type:
        common_head_names.extend(['dense'])
    elif 'electra' in model_type:
        common_head_names.extend(['dense'])
    elif 'deberta' in model_type:
        common_head_names.extend(['pooler'])
    elif 'distilbert' in model_type:
        common_head_names.extend(['pre_classifier'])
    elif any(arch in model_type for arch in ['gpt', 'opt', 'bloom', 'llama', 'mistral']):
        # For decoder-only models, sometimes the classification head is different
        common_head_names.extend(['transformer.ln_f', 'model.norm'])
    
    # Check for exact matches first
    for head_name in common_head_names:
        if hasattr(model, head_name):
            module = getattr(model, head_name)
            classification_heads.append((head_name, module))
    
    # If no exact matches, search through all named modules
    if not classification_heads:
        for name, module in model.named_modules():
            # Look for modules that likely represent classification heads
            if any(head_pattern in name.lower() for head_pattern in ['classif', 'score', 'head', 'output']):
                # Check if it's a linear layer or contains linear layers
                if isinstance(module, torch.nn.Linear) or any(isinstance(m, torch.nn.Linear) for m in module.modules()):
                    classification_heads.append((name, module))
    
    return classification_heads

def detect_encoder_decoder_model(model) -> bool:
    """
    Check if the model is an encoder-decoder architecture.
    
    Args:
        model: The model to check
        
    Returns:
        bool: True if it's an encoder-decoder model
    """
    # Check for common encoder-decoder attributes
    encoder_decoder_indicators = ['encoder', 'decoder', 'shared']
    
    has_encoder = hasattr(model, 'encoder')
    has_decoder = hasattr(model, 'decoder')
    
    # T5, BART, etc. typically have both encoder and decoder
    if has_encoder and has_decoder:
        return True
    
    # Check model type
    model_type = type(model).__name__.lower()
    encoder_decoder_types = ['t5', 'bart', 'pegasus', 'mbart', 'marian']
    
    return any(enc_dec_type in model_type for enc_dec_type in encoder_decoder_types)

def find_linear_layers_by_output_size(model, target_output_size: int) -> List[Tuple[str, torch.nn.Module]]:
    """
    Find linear layers that output to the target size (likely classification heads).
    
    Args:
        model: The model to analyze
        target_output_size: Expected output size (typically num_labels)
        
    Returns:
        List of (name, module) tuples for linear layers with matching output size
    """
    matching_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == target_output_size:
            matching_layers.append((name, module))
    
    return matching_layers

def get_model_architecture_info(model) -> Dict[str, Any]:
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
                
        # Also unfreeze final classification layer weights (optional extension, might depend on model architecture)
        try:
            classifier = model.classifier
            classifier.dense.weight.requires_grad = True
            classifier.out_proj.weight.requires_grad = True
        except AttributeError:
            print("Warning: classifier structure may not match. Inspect model.classifier manually.")
                
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
        
        # Find and unfreeze the classification head
        classification_heads_found = []
        
        # Common classification head names by model family (AutoModelForSequenceClassification guarantees these exist)
        classification_head_patterns = {
            'bert': ['classifier'],
            'roberta': ['classifier'], 
            'electra': ['classifier'],
            'deberta': ['classifier'],
            'distilbert': ['classifier'],
            'llama': ['score'],
            'gpt': ['score'],
            'opt': ['score'], 
            'bloom': ['score'],
            'mistral': ['score'],
            'qwen': ['score'],
            'gemma': ['score'],
            # Add more as needed
        }
        
        # Determine model family and expected head names
        model_family = None
        expected_heads = []
        
        for family, heads in classification_head_patterns.items():
            if family in model_type.lower():
                model_family = family
                expected_heads = heads
                break
        
        # If we don't recognize the family, use common fallbacks
        if not expected_heads:
            expected_heads = ['classifier', 'score', 'head']
            logger.debug(f"Unknown model family for {model_type}, using common head names: {expected_heads}")
        else:
            logger.debug(f"Detected {model_family} family, looking for heads: {expected_heads}")
        
        # Find and unfreeze classification heads
        for head_name in expected_heads:
            if hasattr(model, head_name):
                head_module = getattr(model, head_name)
                param_count = sum(p.numel() for p in head_module.parameters())
                
                for param in head_module.parameters():
                    param.requires_grad = True
                
                classification_heads_found.append(head_name)
                logger.info(f"Unfroze classification head '{head_name}' with {param_count:,} parameters")
        
        # Validation: We should have found at least one classification head
        if not classification_heads_found:
            # This is very unlikely with AutoModelForSequenceClassification, but let's be safe
            logger.warning("No standard classification heads found. Searching all modules...")
            
            # Look for any linear layer with the correct output size
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.out_features == self.num_labels:
                    param_count = sum(p.numel() for p in module.parameters())
                    for param in module.parameters():
                        param.requires_grad = True
                    classification_heads_found.append(name)
                    logger.info(f"Unfroze linear layer '{name}' with {param_count:,} parameters")
                    break
        
        # Allow additional custom parameters to be unfrozen if specified
        unfrozen_params = self.peft_args.get("unfrozen_params", [])
        custom_unfrozen_count = 0
        
        if unfrozen_params:
            logger.info(f"Attempting to unfreeze {len(unfrozen_params)} custom parameter path(s)")
            
            for param_path in unfrozen_params:
                if set_requires_grad_by_path(model, param_path):
                    custom_unfrozen_count += 1
        
        # Final validation and statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if trainable_params == 0:
            error_msg = (
                f"No trainable parameters found after attempting to unfreeze classification head. "
                f"Model type: {model_type}, Classification heads found: {classification_heads_found}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log final statistics
        percentage = 100 * trainable_params / total_params
        logger.info(f"Linear probing setup complete:")
        logger.info(f"  - Classification heads unfrozen: {classification_heads_found}")
        logger.info(f"  - Custom parameters unfrozen: {custom_unfrozen_count}")
        logger.info(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")
        
        # Optional: Verify the model can still do forward pass
        if self.peft_args.get("validate_forward", True):
            try:
                # Create a small test input
                test_input = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10
                if self.device and torch.cuda.is_available():
                    test_input = test_input.to(self.device)
                    
                with torch.no_grad():
                    output = model(test_input)
                    if hasattr(output, 'logits'):
                        expected_shape = (1, self.num_labels)
                        actual_shape = output.logits.shape
                        if actual_shape == expected_shape:
                            logger.info(f"Forward pass validation successful: output shape {actual_shape}")
                        else:
                            logger.warning(f"Forward pass shape mismatch: expected {expected_shape}, got {actual_shape}")
                    else:
                        logger.warning("Model output doesn't have 'logits' attribute")
            except Exception as e:
                logger.warning(f"Forward pass validation failed: {e}")
        
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
        return model

def validate_classification_setup(model, num_labels: int) -> Dict[str, Any]:
    """
    Validate that the classification setup is correct.
    
    Args:
        model: The model to validate
        num_labels: Expected number of labels
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'trainable_params': 0,
        'total_params': 0,
        'classification_layers': []
    }
    
    try:
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        validation_results['trainable_params'] = trainable_params
        validation_results['total_params'] = total_params
        
        # Check if any parameters are trainable
        if trainable_params == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No trainable parameters found")
        
        # Find classification layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.out_features == num_labels:
                    validation_results['classification_layers'].append({
                        'name': name,
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'trainable': any(p.requires_grad for p in module.parameters())
                    })
        
        # Check if we have the right output size
        classification_outputs = [layer for layer in validation_results['classification_layers'] 
                                if layer['out_features'] == num_labels]
        
        if not classification_outputs:
            validation_results['warnings'].append(
                f"No linear layers found with output size {num_labels}. "
                "This might indicate a configuration mismatch."
            )
        
        # Check for suspiciously small number of trainable parameters
        if trainable_params > 0 and trainable_params < 1000:
            validation_results['warnings'].append(
                f"Very few trainable parameters ({trainable_params}). "
                "This might indicate incomplete unfreezing."
            )
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation failed: {str(e)}")
    
    return validation_results

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
    return factory
