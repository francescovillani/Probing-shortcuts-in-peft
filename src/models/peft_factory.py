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
    PeftModel,
    PeftConfig,
)
# Optional: Adapter-Transformers (AdapterHub) for Pfeiffer adapters
try:
    import adapters
    from adapters import AdapterConfig
    ADAPTERS_AVAILABLE = True
except Exception:
    ADAPTERS_AVAILABLE = False
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def set_requires_grad_by_path(model, param_path: str) -> bool:
    """
    Set requires_grad=True for a parameter at the given path.

    Args:
        model: The model to modify
        param_path: Dot-separated path to the parameter (e.g., 'classifier.dense.weight')

    Returns:
        bool: True if parameter was found and modified, False otherwise
    """
    parts = param_path.split(".")
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
            "name": name,
            "shape": list(param.shape),
            "count": param_count,
            "dtype": str(param.dtype),
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
            logger.debug(
                f"  {param_info['name']}: {param_info['shape']} ({param_info['count']:,} params, {param_info['dtype']})"
            )
    else:
        logger.debug("No trainable parameters found!")

    # Log a summary of frozen parameter types if there are many
    if len(frozen_params) > 20:
        logger.debug(
            f"\nFrozen parameters: {len(frozen_params)} layers (showing summary)"
        )
        # Group by parameter type
        frozen_summary = {}
        for param_info in frozen_params:
            param_type = param_info["name"].split(".")[
                -1
            ]  # Get the last part (weight, bias, etc.)
            if param_type not in frozen_summary:
                frozen_summary[param_type] = {"count": 0, "total_params": 0}
            frozen_summary[param_type]["count"] += 1
            frozen_summary[param_type]["total_params"] += param_info["count"]

        for param_type, summary in frozen_summary.items():
            logger.debug(
                f"  {param_type}: {summary['count']} layers, {summary['total_params']:,} params"
            )
    elif frozen_params:
        logger.debug(f"\nFrozen parameters ({len(frozen_params)} layers):")
        for param_info in frozen_params[:10]:  # Show first 10 to avoid spam
            logger.debug(
                f"  {param_info['name']}: {param_info['shape']} ({param_info['count']:,} params)"
            )
        if len(frozen_params) > 10:
            logger.debug(f"  ... and {len(frozen_params) - 10} more frozen layers")

    logger.debug("=" * 50)


def unfreeze_classification_head(base_model, num_labels, peft_args, method_name):
    """
    Unfreeze the classification head and/or specific parameters as requested via peft_args.

    This is critical because the classification head is randomly initialized when using
    AutoModelForSequenceClassification and needs to be trained.

    Args:
        base_model: The base model with the classification head
        num_labels: Expected number of output labels (currently unused)
        peft_args: Dict that may contain:
            - 'heads_to_save': list of classification head module names to unfreeze and later save.
              Only top-level module names are supported for save/load compatibility.
            - 'unfrozen_params': list of dotted parameter paths to set requires_grad=True (e.g.,
              'classifier.dense.weight'). These are not considered classification heads to save.
        method_name: Name of the PEFT method for logging

    Returns:
        classification_heads_found: List[str] of TOP-LEVEL classification head module names found
            in the model and un-frozen. This list is intended to drive save/load of head weights.
    """
    classification_heads_found = []

    heads_to_save = list(peft_args.get("heads_to_save", []))
    param_paths_to_unfreeze = list(peft_args.get("unfrozen_params", []))

    # Handle heads_to_save: unfreeze entire (top-level) modules and track for saving
    if heads_to_save:
        logger.info(
            f"[{method_name}] User specified heads to unfreeze: {heads_to_save}"
        )
        for head_name in heads_to_save:
            # Normalize to top-level attribute for save/load compatibility
            top_level_name = head_name.split(".")[0]
            if head_name != top_level_name:
                logger.warning(
                    f"[{method_name}] Specified head '{head_name}' includes a nested path. "
                    f"Using top-level module '{top_level_name}' for save/load compatibility."
                )

            if hasattr(base_model, top_level_name):
                head_module = getattr(base_model, top_level_name)

                # If we actually resolved a module, unfreeze all its parameters
                if hasattr(head_module, "parameters"):
                    param_count = sum(p.numel() for p in head_module.parameters())
                    for param in head_module.parameters():
                        param.requires_grad = True
                    if top_level_name not in classification_heads_found:
                        classification_heads_found.append(top_level_name)
                    logger.info(
                        f"[{method_name}] Unfroze head module '{top_level_name}' with {param_count:,} parameters"
                    )
                else:
                    # Fallback: if it's not a module, try treating the original head_name as a parameter path
                    # This is unusual for heads_to_save, but we attempt to honor the intent
                    success = set_requires_grad_by_path(base_model, head_name)
                    if not success:
                        logger.warning(
                            f"[{method_name}] Could not unfreeze '{head_name}' as module or parameter path."
                        )
            else:
                logger.warning(
                    f"[{method_name}] Specified head module '{top_level_name}' not found in the model."
                )

    # Handle unfrozen_params: set requires_grad=True on arbitrary parameter paths
    if param_paths_to_unfreeze:
        logger.info(
            f"[{method_name}] User specified parameter paths to unfreeze: {param_paths_to_unfreeze}"
        )
        for param_path in param_paths_to_unfreeze:
            # Avoid re-processing entries that were already covered by heads_to_save
            if param_path.split(".")[0] in classification_heads_found:
                continue
            _ = set_requires_grad_by_path(base_model, param_path)

    return classification_heads_found


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
            self.model_name,
            num_labels=self.num_labels,
            quantization_config=quantization_config,
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

        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        config = PromptTuningConfig(**peft_config_args)
        model = get_peft_model(base_model, config)

        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info(
            f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )

        if self.device:
            model = model.to(self.device)
        return model


# Prefix Tuning factory
class PrefixTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        config = PrefixTuningConfig(**peft_config_args)
        model = get_peft_model(base_model, config)

        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(
            f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )

        if self.device:
            model = model.to(self.device)
        return model


# P-Tuning v2 factory
class PTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        # Create PEFT config and apply it
        peft_config_args = self.peft_args.copy()
        config = PromptEncoderConfig(**peft_config_args)
        model = get_peft_model(base_model, config)

        # Log statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info(
            f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )

        if self.device:
            model = model.to(self.device)
        return model


# Pfeiffer Adapters (AdapterHub) factory
class PfeifferAdapterModelFactory(PEFTModelFactory):
    def create_model(self):
        if not ADAPTERS_AVAILABLE:
            raise ImportError(
                "transformers.adapters not available. Install Adapter-Transformers: pip install adapter-transformers"
            )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        # Adapter configuration and name
        peft_args = self.peft_args.copy()
        adapter_name = peft_args.pop("adapter_name", "pfeiffer")


        # Add and activate adapter
        adapters.init(base_model)
        base_model.add_adapter(adapter_name, config="pfeiffer", set_active=True)
        base_model.train_adapter(adapter_name)

        # Remember adapter name for saving
        try:
            setattr(base_model, "trained_adapter_name", adapter_name)
        except Exception:
            pass

        # Optionally unfreeze classification head(s)
        classification_heads_found = (
            unfreeze_classification_head(
                base_model, self.num_labels, self.peft_args, "Pfeiffer-Adapter"
            )
        )
        if classification_heads_found:
            base_model.classification_heads_to_save = classification_heads_found

        if self.device:
            base_model = base_model.to(self.device)
        return base_model

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

        classification_heads_found = (
            unfreeze_classification_head(
                model, self.num_labels, self.peft_args, "BitFit"
            )
        )
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
            
        # Log statistics
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(
            f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )

        if self.device:
            model = model.to(self.device)
        return model


# LayerNorm Tuning factory
class LayerNormTuningModelFactory(PEFTModelFactory):
    def create_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm parameters
        layernorm_param_count = 0
        layernorm_modules_found = 0
        
        for name, module in model.named_modules():
            # Check for LayerNorm modules (works for both nn.LayerNorm and model-specific LayerNorm classes)
            if isinstance(module, torch.nn.LayerNorm) or 'LayerNorm' in type(module).__name__:
                layernorm_modules_found += 1
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    layernorm_param_count += param.numel()
                    logger.debug(f"Unfroze LayerNorm parameter: {name}.{param_name} ({param.numel():,} params)")

        logger.info(f"Found {layernorm_modules_found} LayerNorm modules with {layernorm_param_count:,} total parameters")

        classification_heads_found = (
            unfreeze_classification_head(
                model, self.num_labels, self.peft_args, "LayerNorm-Tuning"
            )
        )
        if classification_heads_found:
            model.classification_heads_to_save = classification_heads_found
            
        # Log statistics
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        logger.info(
            f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)"
        )
        logger.info(
            f"  - LayerNorm parameters: {layernorm_param_count:,} ({100 * layernorm_param_count / total_params:.4f}%)"
        )

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
        config_type = (
            type(model.config).__name__ if hasattr(model, "config") else "Unknown"
        )
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(f"Model architecture: {model_type} ({config_type})")
        logger.debug(f"Model has {total_params:,} total parameters")

        # Since we use AutoModelForSequenceClassification, we're guaranteed to have a proper classification head
        # The head will be named according to the model family's convention

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        classification_heads_found = (
            unfreeze_classification_head(
                model, self.num_labels, self.peft_args, "Classifier-only"
            )
        )
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
                torch_dtype=(
                    config.torch_dtype
                    if hasattr(config, "torch_dtype")
                    else torch.float32
                ),
            )
            model = PeftModel.from_pretrained(
                base_model, peft_model_path, is_trainable=False
            )
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

            with open(custom_config_path, "r") as f:
                custom_config = json.load(f)

            head_names = custom_config.get("classification_heads_to_save", [])
            if head_names:
                model.classification_heads_to_save = head_names
                # The base model holds the classification head
                base_model = model.base_model if hasattr(model, "base_model") else model

                for head_name in head_names:
                    classifier_weights_path = Path(peft_model_path) / f"{head_name}.pt"
                    if not classifier_weights_path.exists():
                        logger.warning(
                            f"Could not find weights file for classification head '{head_name}' at {classifier_weights_path}"
                        )
                        continue

                    if hasattr(base_model, head_name):
                        head_module = getattr(base_model, head_name)
                        logger.info(
                            f"Loading custom classification head '{head_name}' from {classifier_weights_path}"
                        )
                        state_dict = torch.load(
                            classifier_weights_path, map_location=self.device
                        )
                        head_module.load_state_dict(state_dict)
                    else:
                        logger.warning(
                            f"Could not find classification head '{head_name}' in the model."
                        )

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
    "layernorm_tuning": LayerNormTuningModelFactory,
    "classifier_only": ClassifierOnlyModelFactory,
    "load_peft": LoadPeftModelFactory,
    "pfeiffer_adapter": PfeifferAdapterModelFactory,
}


def get_peft_model_factory(
    peft_type, model_name, num_labels, peft_args=None, device=None
):

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
