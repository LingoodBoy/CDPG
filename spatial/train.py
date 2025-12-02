import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import create_dataloaders
from models.clip_model import CLIPModel
from loss import InfoNCELoss


class Trainer:

    def __init__(self, config: dict, resume_from: str = None):

        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() or config['device'] == 'mps' else 'cpu')
        print(f" Using device: {self.device}")

        torch.manual_seed(config['training']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['training']['seed'])

        self.model = CLIPModel(config).to(self.device)
        print(f"Model loaded to {self.device}")

        self.criterion = InfoNCELoss()

        print("\nLoading datasets...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config,
            batch_size=config['training']['phase1']['batch_size'],
            num_workers=config['training']['num_workers']
        )

        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir / timestamp)

        self.global_step = 0
        self.current_epoch = 0
        self.current_phase = 1
        self.best_val_loss = float('inf')

        if resume_from:
            self.load_checkpoint(resume_from)

    def train_phase(self, phase: int):

        self.current_phase = phase
        phase_config_key = f'phase{phase}'

        if phase == 3 and not self.config['training']['phase3'].get('enabled', False):
            print(f"\nPhase 3 is disabled, skipping...")
            return

        phase_config = self.config['training'][phase_config_key]

        print(f"\n{'='*70}")
        print(f"Starting Phase {phase}")
        print(f"{'='*70}")

        self.model.configure_phase(phase, self.config)
        self.model.print_trainable_params()

        optimizer = self._create_optimizer(phase)

        num_epochs = phase_config['epochs']
        for epoch in range(num_epochs):
            self.current_epoch += 1

            print(f"\nEpoch {self.current_epoch} (Phase {phase}, {epoch+1}/{num_epochs})")

            train_metrics = self.train_epoch(optimizer)

            val_metrics = self.validate()

            self._print_metrics(train_metrics, val_metrics)

            self._log_metrics(train_metrics, val_metrics)

            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'phase{phase}_epoch{self.current_epoch}.pt')

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                print(f"Best model saved (val_loss: {val_metrics['loss']:.4f})")

        print(f"\nPhase {phase} completed!")

    def train_epoch(self, optimizer: optim.Optimizer) -> dict:
        self.model.train()

        total_loss = 0
        total_metrics = {
            'loss_i2t': 0,
            'loss_t2i': 0,
            'i2t_acc': 0,
            't2i_acc': 0,
            'avg_acc': 0
        }

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            image_embeddings, text_embeddings = self.model(images, input_ids, attention_mask)

            loss, metrics = self.criterion(
                image_embeddings,
                text_embeddings,
                temperature=self.model.temperature.item()
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )

            optimizer.step()

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['avg_acc']:.4f}"
            })

            self.global_step += 1

            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/Loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy_step', metrics['avg_acc'], self.global_step)

        num_batches = len(self.train_loader)
        avg_metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> dict:

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

    def _create_optimizer(self, phase: int) -> optim.Optimizer:

        phase_config = self.config['training'][f'phase{phase}']

        if phase == 1:

            params = [
                {'params': self.model.image_encoder.projection.parameters()},
                {'params': self.model.text_encoder.projection.parameters()},
                {'params': [self.model.temperature]}
            ]
            lr = phase_config['learning_rate']
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.config['training']['weight_decay'])

        else:

            lr_backbone = phase_config['learning_rate_backbone']
            lr_projection = phase_config['learning_rate_projection']

            params = [
                {'params': self.model.image_encoder.backbone.parameters(), 'lr': lr_backbone},
                {'params': self.model.image_encoder.projection.parameters(), 'lr': lr_projection},
                {'params': self.model.text_encoder.bert.parameters(), 'lr': lr_backbone},
                {'params': self.model.text_encoder.projection.parameters(), 'lr': lr_projection},
                {'params': [self.model.temperature], 'lr': lr_projection}
            ]
            optimizer = optim.AdamW(params, weight_decay=self.config['training']['weight_decay'])

        print(f"Optimizer: AdamW")
        if phase == 1:
            print(f"   Learning rate: {lr:.2e}")
        else:
            print(f"   Backbone LR: {lr_backbone:.2e}")
            print(f"   Projection LR: {lr_projection:.2e}")

        return optimizer

    def _print_metrics(self, train_metrics: dict, val_metrics: dict):

        print(f"\nMetrics:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train Acc:  {train_metrics['avg_acc']:.4f} | Val Acc:  {val_metrics['avg_acc']:.4f}")
        print(f"   Val I2T Acc: {val_metrics['i2t_acc']:.4f} | Val T2I Acc: {val_metrics['t2i_acc']:.4f}")
        print(f"   Val I2T R@5: {val_metrics['i2t_recall@5']:.4f} | Val T2I R@5: {val_metrics['t2i_recall@5']:.4f}")

    def _log_metrics(self, train_metrics: dict, val_metrics: dict):

        self.writer.add_scalar('Train/Loss', train_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Train/Accuracy', train_metrics['avg_acc'], self.current_epoch)

        self.writer.add_scalar('Val/Loss', val_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Val/Accuracy', val_metrics['avg_acc'], self.current_epoch)
        self.writer.add_scalar('Val/I2T_Accuracy', val_metrics['i2t_acc'], self.current_epoch)
        self.writer.add_scalar('Val/T2I_Accuracy', val_metrics['t2i_acc'], self.current_epoch)
        self.writer.add_scalar('Val/I2T_Recall@5', val_metrics['i2t_recall@5'], self.current_epoch)
        self.writer.add_scalar('Val/T2I_Recall@5', val_metrics['t2i_recall@5'], self.current_epoch)

        self.writer.add_scalar('Model/Temperature', self.model.temperature.item(), self.current_epoch)

    def save_checkpoint(self, filename: str):

        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str):

        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.current_phase = checkpoint['phase']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Resumed from epoch {self.current_epoch}, phase {self.current_phase}")

    def run(self):

        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)

        self.train_phase(1)

        self.train_phase(2)

        if self.config['training']['phase3'].get('enabled', False):
            self.train_phase(3)

        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)

        print("\n Running final test...")
        test_metrics = self.validate()
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['avg_acc']:.4f}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.yaml', help='')
    parser.add_argument('--resume', type=str, default=None, help='')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, resume_from=args.resume)

    trainer.run()


if __name__ == '__main__':
    main()
