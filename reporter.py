"""
Reporter: logs metrics to Weights & Biases and a local CSV simultaneously.
Designed so experiments can be compared in the W&B dashboard and also
inspected offline without an internet connection.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from config import Config


class Reporter:
    def __init__(self, config: Config, fold: int | None = None):
        self.config = config
        self.fold = fold
        self.exp_dir = config.exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        suffix = f'_fold{fold}' if fold is not None else ''
        self.csv_path = self.exp_dir / f'metrics{suffix}.csv'
        self._csv_header_written = False
        self._fieldnames: list | None = None
        self._best = {'dice_mean': 0.0, 'epoch': 0}

        self._wandb_active = False
        if config.use_wandb:
            run_name = f'exp{config.exp_id}_{config.exp_name}{suffix}'
            try:
                wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config={
                        'exp_id': config.exp_id,
                        'exp_name': config.exp_name,
                        'loss_type': config.loss_type,
                        'encoder': config.encoder_name,
                        'wavelet_family': config.wavelet_family,
                        'wavelet_level': config.wavelet_level,
                        'wavelet_skip_indices': list(config.wavelet_skip_indices),
                        'fold': fold,
                        'batch_size': config.batch_size,
                        'max_lr': config.learning_rate,
                        'scheduler_pct_start': config.scheduler_pct_start,
                        'epochs': config.num_epochs,
                        'dataset_name': config.dataset_name,
                        'allowed_suffixes': list(config.allowed_suffixes) if config.allowed_suffixes else 'all',
                    },
                    reinit='finish_previous',
                )
                self._wandb_active = True
            except Exception as e:
                print(f'[Reporter] W&B init failed ({e}). Logging to CSV only.')
                print('[Reporter] Run `wandb login` to enable W&B tracking.')

    def log(self, split: str, epoch: int, metrics: dict[str, float], lr: float | None = None):
        """Log scalar metrics for one epoch."""
        row = {'split': split, 'epoch': epoch, **metrics}
        if lr is not None:
            row['lr'] = lr

        # CSV
        self._write_csv(row)

        # Track best val Dice
        if split == 'val' and metrics.get('dice_mean', 0) > self._best['dice_mean']:
            self._best = {'dice_mean': metrics['dice_mean'], 'epoch': epoch}

        # W&B
        if self._wandb_active:
            wb_metrics = {f'{split}/{k}': v for k, v in metrics.items()}
            if lr is not None:
                wb_metrics['train/lr'] = lr
            wandb.log(wb_metrics, step=epoch)

    def log_images(
        self,
        images: torch.Tensor,     # [B, 3, H, W] normalized
        masks_gt: torch.Tensor,   # [B, C, H, W]
        logits: torch.Tensor,     # [B, C, H, W]
        epoch: int,
        class_names: tuple[str, ...],
        n: int = 4,
    ):
        """Log segmentation overlays to W&B (called every N epochs)."""
        if not self._wandb_active:
            return

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        preds = (torch.sigmoid(logits) > 0.5).float()   # [B, C, H, W]
        wb_images = []

        for i in range(min(n, images.shape[0])):
            img = (images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)

            masks = {}
            for c, name in enumerate(class_names):
                masks[f'gt_{name}']   = {'mask_data': masks_gt[i, c].cpu().numpy().astype(np.uint8)}
                masks[f'pred_{name}'] = {'mask_data': preds[i, c].cpu().numpy().astype(np.uint8)}

            wb_images.append(wandb.Image(img, masks=masks, caption=f'epoch {epoch} sample {i}'))

        wandb.log({'val/predictions': wb_images}, step=epoch)

    def save_checkpoint(self, model: torch.nn.Module, epoch: int, metrics: dict, tag: str = 'best'):
        path = self.exp_dir / f'model_{tag}_fold{self.fold}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
        }, path)
        return path

    def save_test_results(self, results: dict):
        path = self.exp_dir / 'test_results.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Test results saved → {path}')
        if self._wandb_active:
            wandb.log({'test/' + k: v for k, v in results.items()})

    def finish(self):
        if self._wandb_active:
            wandb.finish()

    @property
    def best_dice(self) -> float:
        return self._best['dice_mean']

    @property
    def best_epoch(self) -> int:
        return self._best['epoch']

    def save_training_plot(self):
        """Save loss + Dice curves to PNG. Called after all epochs of a fold."""
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(self.csv_path)
        train = df[df['split'] == 'train'].set_index('epoch')
        val   = df[df['split'] == 'val'].set_index('epoch')

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        axes[0].plot(train.index, train['loss'], label='train')
        axes[0].plot(val.index,   val['loss'],   label='val')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(train.index, train['dice_mean'], label='train')
        axes[1].plot(val.index,   val['dice_mean'],   label='val')
        best_ep = int(val['dice_mean'].idxmax())
        axes[1].axvline(best_ep, color='gray', linestyle='--', label=f'best ep {best_ep}')
        axes[1].set_ylabel('Dice (mean)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fold_label = f'fold {self.fold}' if self.fold is not None else ''
        fig.suptitle(f'Exp {self.config.exp_id} — {self.config.exp_name}  {fold_label}')
        plt.tight_layout()
        out = self.exp_dir / f'training_curve_fold{self.fold}.png'
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Training curve → {out}')

    def _write_csv(self, row: dict):
        if not self._csv_header_written:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames,
                                        extrasaction='ignore', restval='')
                writer.writeheader()
                writer.writerow(row)
            self._csv_header_written = True
        else:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames,
                                        extrasaction='ignore', restval='')
                writer.writerow(row)
