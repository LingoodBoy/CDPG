import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import create_dataloaders
from models.clip_model import CLIPModel
from loss import InfoNCELoss


class SimpleTrainer:

    def __init__(self, config: dict, resume_from: str = None):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")

        torch.manual_seed(config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training']['seed'])

        self.model = CLIPModel(config).to(self.device)

        self.model.image_encoder.unfreeze_backbone()
        self.model.text_encoder.unfreeze_backbone()

        print(f"Model loaded to {self.device}")
        print(f"Total trainable params: {self._count_params():,}")

        self.criterion = InfoNCELoss()

        print("\nLoading datasets...")
        batch_size = config['training']['batch_size']
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config,
            batch_size=batch_size,
            num_workers=config['training']['num_workers']
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        num_training_steps = len(self.train_loader) * config['training']['epochs']
        num_warmup_steps = len(self.train_loader) * config['training']['warmup_epochs']

        self.scheduler = self._create_scheduler(num_training_steps, num_warmup_steps)

        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("⚡ Mixed precision training enabled")

        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir / timestamp)

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        if resume_from:
            self.load_checkpoint(resume_from)

    def _count_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _create_scheduler(self, num_training_steps, num_warmup_steps):

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:

                return float(current_step) / float(max(1, num_warmup_steps))
            else:

                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        total_metrics = {
            'loss_i2t': 0,
            'loss_t2i': 0,
            'i2t_acc': 0,
            't2i_acc': 0,
            'avg_acc': 0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, batch in enumerate(pbar):

            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            if self.use_amp:
                with autocast():
                    image_embeddings, text_embeddings = self.model(images, input_ids, attention_mask)
                    loss, metrics = self.criterion(
                        image_embeddings,
                        text_embeddings,
                        temperature=self.model.temperature.item()
                    )
            else:
                image_embeddings, text_embeddings = self.model(images, input_ids, attention_mask)
                loss, metrics = self.criterion(
                    image_embeddings,
                    text_embeddings,
                    temperature=self.model.temperature.item()
                )

            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['avg_acc']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            self.global_step += 1

            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/Loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy_step', metrics['avg_acc'], self.global_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.global_step)

        num_batches = len(self.train_loader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0
        total_metrics = {
            'loss_i2t': 0,
            'loss_t2i': 0,
            'i2t_acc': 0,
            't2i_acc': 0,
            'avg_acc': 0,
            'i2t_recall@5': 0,
            't2i_recall@5': 0
        }

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            image_embeddings, text_embeddings = self.model(images, input_ids, attention_mask)
            loss, metrics = self.criterion(
                image_embeddings,
                text_embeddings,
                temperature=self.model.temperature.item()
            )

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

        num_batches = len(self.val_loader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }

        return avg_metrics

    def save_checkpoint(self, filename: str):

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str):

        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed from epoch {self.current_epoch}")

    def run(self):

        print("\n" + "="*70)
        print("Starting Training (Simple Mode - No Phases)")
        print("="*70)
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print("="*70)

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")

            train_metrics = self.train_epoch()

            val_metrics = self.validate()

            print(f"\nMetrics:")
            print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Train Acc:  {train_metrics['avg_acc']:.4f} | Val Acc:  {val_metrics['avg_acc']:.4f}")
            print(f"   Val I2T Acc: {val_metrics['i2t_acc']:.4f} | Val T2I Acc: {val_metrics['t2i_acc']:.4f}")
            print(f"   Val I2T R@5: {val_metrics['i2t_recall@5']:.4f} | Val T2I R@5: {val_metrics['t2i_recall@5']:.4f}")

            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['avg_acc'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['avg_acc'], epoch)
            self.writer.add_scalar('Model/Temperature', self.model.temperature.item(), epoch)

            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                print(f"Best model saved (val_loss: {val_metrics['loss']:.4f})")

        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='all')
    parser.add_argument('--config', type=str, default='config_simple.yaml', help='config path')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SimpleTrainer(config, resume_from=args.resume)

    trainer.run()


if __name__ == '__main__':
    main()
