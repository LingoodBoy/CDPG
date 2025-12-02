from __future__ import annotations
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = image_embeddings.shape[0]

        similarity = image_embeddings @ text_embeddings.T / temperature

        labels = torch.arange(batch_size, device=image_embeddings.device)

        loss_i2t = self.cross_entropy(similarity, labels)

        loss_t2i = self.cross_entropy(similarity.T, labels)

        loss = (loss_i2t + loss_t2i) / 2

        with torch.no_grad():

            i2t_preds = similarity.argmax(dim=1)
            i2t_acc = (i2t_preds == labels).float().mean()

            t2i_preds = similarity.T.argmax(dim=1)
            t2i_acc = (t2i_preds == labels).float().mean()

            i2t_recall_at_5 = self._recall_at_k(similarity, labels, k=5)
            t2i_recall_at_5 = self._recall_at_k(similarity.T, labels, k=5)

        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'i2t_acc': i2t_acc.item(),
            't2i_acc': t2i_acc.item(),
            'i2t_recall@5': i2t_recall_at_5,
            't2i_recall@5': t2i_recall_at_5,
            'avg_acc': ((i2t_acc + t2i_acc) / 2).item()
        }

        return loss, metrics

    @staticmethod
    def _recall_at_k(similarity: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:

        _, top_k_indices = similarity.topk(k, dim=1)
        labels_expanded = labels.unsqueeze(1).expand(-1, k)
        recall = (top_k_indices == labels_expanded).any(dim=1).float().mean().item()
        return recall


class CLIPLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = image_embeddings.shape[0]

        # Cosine similarity 
        logits = image_embeddings @ text_embeddings.T / temperature  # [B, B]

        labels = torch.arange(batch_size, device=image_embeddings.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.T.argmax(dim=1) == labels).float().mean()

        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i.item(),
            'loss_t2i': loss_t.item(),
            'i2t_acc': i2t_acc.item(),
            't2i_acc': t2i_acc.item(),
            'avg_acc': ((i2t_acc + t2i_acc) / 2).item()
        }

        return loss, metrics


if __name__ == '__main__':
    print("\nTesting InfoNCE Loss...")

    batch_size = 8
    embedding_dim = 64

    image_embeddings = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
    text_embeddings = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)

    for i in range(batch_size):
        text_embeddings[i] = 0.8 * text_embeddings[i] + 0.2 * image_embeddings[i]
        text_embeddings[i] = F.normalize(text_embeddings[i], dim=-1)

    criterion = InfoNCELoss()
    loss, metrics = criterion(image_embeddings, text_embeddings, temperature=0.07)

    print(f"Loss: {loss.item():.4f}")
    print(f"Image-to-Text Loss: {metrics['loss_i2t']:.4f}")
    print(f"Text-to-Image Loss: {metrics['loss_t2i']:.4f}")
    print(f"Image-to-Text Accuracy: {metrics['i2t_acc']:.4f}")
    print(f"Text-to-Image Accuracy: {metrics['t2i_acc']:.4f}")
    print(f"Image-to-Text Recall@5: {metrics['i2t_recall@5']:.4f}")
    print(f"Text-to-Image Recall@5: {metrics['t2i_recall@5']:.4f}")

    print("\nTesting CLIP Loss...")
    clip_criterion = CLIPLoss()
    clip_loss, clip_metrics = clip_criterion(image_embeddings, text_embeddings, temperature=0.07)

    print(f"CLIP Loss: {clip_loss.item():.4f}")
    print(f"Difference from InfoNCE: {abs(loss.item() - clip_loss.item()):.6f} (should be ~0)")

    print("\nTesting with random embeddings...")
    random_images = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
    random_texts = F.normalize(torch.randn(batch_size, embedding_dim), dim=-1)

    random_loss, random_metrics = criterion(random_images, random_texts)
    print(f"Random Loss: {random_loss.item():.4f} (should be higher)")
    print(f"Random Accuracy: {random_metrics['avg_acc']:.4f} (should be ~1/batch_size = {1/batch_size:.4f})")

    print("\nLoss function test passed!")
