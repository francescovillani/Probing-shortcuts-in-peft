import wandb
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from typing import Optional, Callable, Dict, Any

class TrainingService:
    def __init__(self, device: torch.device):
        self.device = device
        # GradScaler solo per FP16 su CUDA; BF16 e CPU non usano scaler
        use_scaler = (device.type == "cuda" and not torch.cuda.is_bf16_supported())
        self.scaler = GradScaler(enabled=use_scaler)

    def _autocast_kwargs(self):
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return dict(device_type="cuda", dtype=dtype)
        else:  # cpu: autocast supporta solo bfloat16
            return dict(device_type="cpu", dtype=torch.bfloat16)

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        gradient_accumulation_steps: int,
        epoch_num: int,
        privacy_engine=None,
    ) -> float:

        model.train()
        total_loss = 0.0
        num_steps = 0
        use_privacy = privacy_engine is not None

        if use_privacy:
            # === DP PATH: uno step per batch dal dataloader ===
            loader = tqdm(dataloader, desc=f"Epoch {epoch_num + 1} - DP Training")
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                # Nota: con Opacus, niente AMP mixed-precision classica
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()          # DPOptimizer: clip + rumore + update
                lr_scheduler.step()       # step-based: avanza ad ogni update

                total_loss += loss.item()
                num_steps += 1

                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0]
                    })
                loader.set_postfix({"loss": total_loss / num_steps})

        else:
            # === NON-DP PATH: gradient accumulation + AMP (se disponibile) ===
            loader = tqdm(dataloader, desc=f"Epoch {epoch_num + 1} - Training")
            optimizer.zero_grad(set_to_none=True)
            autocast_kwargs = self._autocast_kwargs()

            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(**autocast_kwargs):
                    outputs = model(**batch)
                    raw_loss = outputs.loss

                # gestisce l'ultimo gruppo incompleto
                if gradient_accumulation_steps > 1:
                    remaining = len(dataloader) - batch_idx
                    effective_steps = min(gradient_accumulation_steps, remaining)
                else:
                    effective_steps = 1

                loss = raw_loss / effective_steps
                self.scaler.scale(loss).backward()

                do_step = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if do_step:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()   # step-based: SOLO quando aggiorni davvero

                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": raw_loss.item(),  # loss “grezza” per batch
                            "learning_rate": lr_scheduler.get_last_lr()[0]
                        })
                else:
                    if wandb.run is not None:
                        wandb.log({"train/loss_micro": raw_loss.item()})

                total_loss += raw_loss.item()
                num_steps += 1
                loader.set_postfix({"loss": total_loss / num_steps})

        # media semplice sulle iterazioni del dataloader
        return total_loss / max(1, num_steps)
    
    def train_epoch_with_reweighting(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        gradient_accumulation_steps: int,
        epoch_num: int,
        current_step: int,
        total_steps: int,
        custom_loss_fn: Optional[Callable] = None,
        custom_loss_kwargs: Optional[Dict[str, Any]] = None,
        privacy_engine=None,
    ) -> float:
        """
        Train for one epoch with optional custom loss computation (e.g., for self-debiasing).
        
        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            gradient_accumulation_steps: Number of gradient accumulation steps
            epoch_num: Current epoch number
            current_step: Current training step (for annealing)
            total_steps: Total training steps (for annealing)
            custom_loss_fn: Optional function to compute custom loss
            custom_loss_kwargs: Optional kwargs for custom loss function
            privacy_engine: Optional differential privacy engine
            
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_steps = 0
        use_privacy = privacy_engine is not None

        if use_privacy:
            # === DP PATH: uno step per batch dal dataloader ===
            loader = tqdm(dataloader, desc=f"Epoch {epoch_num + 1} - DP Training")
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                # Nota: con Opacus, niente AMP mixed-precision classica
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()          # DPOptimizer: clip + rumore + update
                lr_scheduler.step()       # step-based: avanza ad ogni update

                total_loss += loss.item()
                num_steps += 1

                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0]
                    })
                loader.set_postfix({"loss": total_loss / num_steps})

        else:
            # === NON-DP PATH: gradient accumulation + AMP (se disponibile) ===
            loader = tqdm(dataloader, desc=f"Epoch {epoch_num + 1} - Training")
            optimizer.zero_grad(set_to_none=True)
            autocast_kwargs = self._autocast_kwargs()

            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(**autocast_kwargs):
                    outputs = model(**batch)
                    raw_loss = outputs.loss

                # gestisce l'ultimo gruppo incompleto
                if gradient_accumulation_steps > 1:
                    remaining = len(dataloader) - batch_idx
                    effective_steps = min(gradient_accumulation_steps, remaining)
                else:
                    effective_steps = 1

                loss = raw_loss / effective_steps
                self.scaler.scale(loss).backward()

                do_step = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader))
                if do_step:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()   # step-based: SOLO quando aggiorni davvero

                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": raw_loss.item(),  # loss “grezza” per batch
                            "learning_rate": lr_scheduler.get_last_lr()[0]
                        })
                else:
                    if wandb.run is not None:
                        wandb.log({"train/loss_micro": raw_loss.item()})

                total_loss += raw_loss.item()
                num_steps += 1
                loader.set_postfix({"loss": total_loss / num_steps})

        # media semplice sulle iterazioni del dataloader
        return total_loss / max(1, num_steps)
