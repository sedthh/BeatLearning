import os
from datetime import datetime
import numpy as np
from tqdm import tqdm

from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .models import BEaRT
from .utils import BEaRTDataset
from .configs import TrainingConfig


class BEaRTTrainer:

    def __init__(self, 
                 name: str, 
                 model: BEaRT,
                 train_dataset: BEaRTDataset,
                 test_dataset: BEaRTDataset,
                 config: TrainingConfig):
        self.name = name
        self.log_name = f"{self.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config
        if self.config.model_directory is not None:
            os.makedirs(self.config.model_directory, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.config.use_cuda_if_available else "cpu")
        self.model = model
        self.model.to(self.device)
        self.optimizer = self._adamw()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay)
        torch.manual_seed(self.model.tokenizer.config.random_seed)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size_train, shuffle=True, num_workers=self.config.num_workers)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size_test, shuffle=False, num_workers=self.config.num_workers)
        self.summary_writer = None
        self.metric_cache = {}
        self.best = {}
        self.patience = 0
    
    @property
    def is_cuda_available(self) -> bool:
        return "cuda" in str(self.device)

    def _adamw(self) -> torch.optim.AdamW:
        # loosely based on https://github.com/karpathy/minGPT/
        decay, nodecay = set(), set()
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param
            if not param.requires_grad:
                nodecay.add(name)
            elif "_embedding." in name:
                if self.config.weight_decay_embeddings:
                    decay.add(name)
                else:
                    nodecay.add(name)
            elif "_head." in name:
                nodecay.add(name)
            elif ".norm" in name:
                nodecay.add(name)
            elif name.endswith("bias"):
                nodecay.add(name)
            else:
                decay.add(name)

        optim_groups = [
            {'params': [params[p] for p in sorted(list(decay))], 'weight_decay': self.config.weight_decay},
            {'params': [params[p] for p in sorted(list(nodecay))], 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.adamw_beta1, self.config.adamw_beta2), fused=self.is_cuda_available)
       
    def save(self, model_path: str, epoch: int = 0):
        torch.save({
            "epoch": epoch,
            "name": self.name,
            "log_name": self.log_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best": {key: value for key, value in self.best.items()},
            "patience": self.patience,
        }, model_path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        if "name" in checkpoint:
            self.name = checkpoint["name"]
            self.log_name = checkpoint["log_name"]
            self.summary_writer = None
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best = {key: value for key, value in checkpoint["best"].items()}
        self.patience = checkpoint["patience"]
        return checkpoint["epoch"]
    
    def metric(self, stuff: Dict[str, float]) -> None:
        for key, value in stuff.items():
            if key not in self.metric_cache:
                self.metric_cache[key] = []
            self.metric_cache[key].append(value)

    def dump(self, n_iter: int) -> Dict[str, float]:
        if self.summary_writer is None and self.config.tensorboard_directory is not None:
            # init TB logs only after the first call
            os.makedirs(self.config.tensorboard_directory, exist_ok=True)
            self.summary_writer = SummaryWriter(os.path.join(self.config.tensorboard_directory, self.log_name))
        results = {key: np.mean(value) for key, value in self.metric_cache.items()}
        if results and self.summary_writer is not None:
            with self.summary_writer as w:
                for key, value in results.items():
                    w.add_scalar(key, value, n_iter)
        self.metric_cache = {}
        return results

    def train(self, epoch: int) -> None:
        self.model.train()
        self.iteration(epoch, self.train_dataloader)

    def test(self, epoch: int) -> None:
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch:int, dataloader: DataLoader, train: bool = True):
        if self.config.early_stopping_rounds is not None and self.patience > self.config.early_stopping_rounds:
            return
        prefix = "train" if train else "test"
        for i, batch in tqdm(enumerate(dataloader), 
                             desc=f"{epoch + 1} / {self.config.num_epochs}",
                             unit=f" {prefix: <5} batch", 
                             total=len(dataloader)):
             
            regression, classification              = self.model(batch["input_data"].to(self.device),
                                                                 batch["segment_data"].to(self.device), 
                                                                 batch["input_audio"].to(self.device))
            loss_regression, loss_classification    = self.model.hearing_loss(regression, 
                                                                              classification, 
                                                                              output_data=batch["output_data"].to(self.device), 
                                                                              output_mask=batch["output_mask"].to(self.device), 
                                                                              tempo=batch["tempo"].float().to(self.device))
            loss = self.config.regression_loss_weight * loss_regression + self.config.classification_loss_weight * loss_classification
            
            if train:
                if not torch.all(torch.isfinite(loss)):
                    print(batch)
                    raise ValueError("NaNs encountered during training!")
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
                self.optimizer.step()
            
            self.metric({
                f"{prefix}/loss": loss.item(), 
                f"{prefix}/loss_reg": loss_regression.item(),
                f"{prefix}/loss_class": loss_classification.item()
            })
            if train and i % self.config.log_frequency == self.config.log_frequency - 1:
                self.dump(epoch * len(dataloader) + i + 1)
        
        if train:
            self.dump(epoch * len(dataloader) + i + 1)
            self.scheduler.step()
        else:
            current = self.dump(epoch)
            if self.config.warmp_up_rounds is not None and epoch < self.config.warmp_up_rounds:
                print(f"Skipping comparison during warmup...")
                return
            if not self.best or current["test/loss"] <= self.best["test/loss"]:
                if epoch > 0:
                    print(f"Model improved from {self.best['test/loss']} to {current['test/loss']}")
                else:
                    print(f"Loss after first epoch: {current['test/loss']}")
                self.best = {key: value for key, value in current.items()}
                if self.config.model_directory is not None:
                    model_path = os.path.join(self.config.model_directory, f"{self.log_name}_{epoch + 1}.pt")
                    print(f"Saving model file: {model_path}")
                    self.save(model_path, epoch)
                self.patience = 0
            else:
                self.patience += 1
                print(f"No improvement for {self.patience} round(s) from {self.best['test/loss']} (current loss is {current['test/loss']})")
                if self.config.early_stopping_rounds is not None and self.patience > self.config.early_stopping_rounds:
                    print(f"Early stopping after no improvements for {self.patience} epochs.")
