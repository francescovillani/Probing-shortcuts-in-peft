import wandb
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

class TrainingService:
    """
    Service responsible for handling the training loop of a model.
    """
    def __init__(self, device: torch.device):
        """
        Initializes the TrainingService.
        Args:
            device: The device to run training on (e.g., 'cuda' or 'cpu').
        """
        self.device = device

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
        """
        Runs one full training epoch.
        Args:
            model: The model to be trained.
            dataloader: DataLoader providing the training data.
            optimizer: The optimizer for updating model weights.
            lr_scheduler: The learning rate scheduler.
            gradient_accumulation_steps: The number of steps to accumulate gradients over.
            epoch_num: The current epoch number (for logging).
            privacy_engine: Opacus privacy engine for differential privacy (optional).
        Returns:
            The average training loss for the epoch.
        """
        model.train()
        total_loss = 0
        train_loader = tqdm(dataloader, desc=f"Epoch {epoch_num + 1} - Training")

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            train_loader.set_postfix({"loss": loss.item()})

            if wandb.run is not None:
                wandb.log({"train/loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]})

        return total_loss / len(dataloader)
