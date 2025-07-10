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
from typing import Optional, Dict, Any
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

def set_requires_grad_by_path(model, param_path):
    parts = param_path.split('.')
    obj = model
    try:
        for part in parts:
            obj = getattr(obj, part)
        obj.requires_grad = True
    except AttributeError:
        print(f"Warning: Could not set requires_grad for {param_path}")

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
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the classification head
        # This handles various model architectures (BERT, RoBERTa, DistilBERT, etc.)
        if hasattr(model, 'classifier'):
            # For most BERT-based models
            for param in model.classifier.parameters():
                param.requires_grad = True
            logger.info("Unfroze classifier parameters")
        elif hasattr(model, 'score'):
            # For some models like RoBERTa variants
            for param in model.score.parameters():
                param.requires_grad = True
            logger.info("Unfroze score parameters")
        elif hasattr(model, 'cls'):
            # For models with cls head
            for param in model.cls.parameters():
                param.requires_grad = True
            logger.info("Unfroze cls parameters")
        else:
            # Fallback: try to find the last linear layer
            last_linear = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    last_linear = module
            
            if last_linear is not None:
                for param in last_linear.parameters():
                    param.requires_grad = True
                logger.info("Unfroze last linear layer parameters")
            else:
                logger.warning("Could not identify classification head. All parameters remain frozen.")
        
        # Allow additional custom parameters to be unfrozen if specified
        unfrozen_params = self.peft_args.get("unfrozen_params", [])
        for param_path in unfrozen_params:
            set_requires_grad_by_path(model, param_path)
        
        # Log trainable parameters count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
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
