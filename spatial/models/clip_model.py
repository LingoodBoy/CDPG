import torch
import torch.nn as nn
from typing import Dict, Tuple

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # Image Encoder
        image_config = config['model']['image_encoder']
        self.image_encoder = ImageEncoder(
            backbone=image_config['backbone'],
            pretrained=image_config['pretrained'],
            image_size=image_config['image_size'],
            embedding_dim=image_config['embedding_dim'],
            dropout=image_config['dropout'],
            freeze=True 
        )

        # Text Encoder
        text_config = config['model']['text_encoder']
        local_files_only = config['model'].get('local_files_only', False)

        self.text_encoder = TextEncoder(
            backbone=text_config['backbone'],
            embedding_dim=text_config['embedding_dim'],
            dropout=text_config['dropout'],
            freeze=True, 
            local_files_only=local_files_only  
        )

        self.temperature = nn.Parameter(
            torch.ones([]) * config['model']['temperature']
        )

        print(f"CLIP Model initialized")
        print(f"   Temperature: {self.temperature.item():.4f}")

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)

        return image_embeddings, text_embeddings

    def get_similarity_matrix(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        similarity = image_embeddings @ text_embeddings.T

        # Temperature scaling
        similarity = similarity / self.temperature

        return similarity

    def configure_phase(self, phase: int, config: dict):
        if phase == 1:
            # Phase 1
            self.image_encoder.freeze_backbone()
            self.text_encoder.freeze_backbone()
            print("\nPhase 1: Encoders frozen, only training projection heads")

        elif phase == 2:
            # Phase 2
            image_unfreeze = config['training']['phase2']['image_unfreeze_blocks']
            text_unfreeze = config['training']['phase2']['text_unfreeze_layers']

            self.image_encoder.unfreeze_last_n_blocks(image_unfreeze)
            self.text_encoder.unfreeze_last_n_layers(text_unfreeze)
            print(f"\nPhase 2: Gradual unfreezing")
            print(f"   Image: last {image_unfreeze} blocks")
            print(f"   Text: last {text_unfreeze} layers")

        elif phase == 3:
            # Phase 3
            self.image_encoder.unfreeze_backbone()
            self.text_encoder.unfreeze_backbone()
            print("\n Phase 3: Full fine-tuning")

        else:
            raise ValueError(f"Invalid phase: {phase}, must be 1, 2, or 3")

    def get_trainable_params(self) -> Dict[str, int]:
        return {
            'image_encoder': self.image_encoder.get_trainable_params(),
            'text_encoder': self.text_encoder.get_trainable_params(),
            'temperature': 1,
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def print_trainable_params(self):
        params = self.get_trainable_params()
        print("\n Trainable Parameters:")
        print(f"   Image Encoder: {params['image_encoder']:,}")
        print(f"   Text Encoder:  {params['text_encoder']:,}")
        print(f"   Temperature:   {params['temperature']:,}")
        print(f"   Total:         {params['total']:,}")


if __name__ == '__main__':
    print("\nTesting CLIP Model...")

    import yaml
    from transformers import AutoTokenizer

    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = CLIPModel(config)
    model.print_trainable_params()

    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = [
        "This image shows a dense mixed-use urban area.",
        "A grid-like road network with a major highway.",
        "Green spaces and parks scattered throughout.",
        "Industrial zones with large buildings."
    ]
    encoding = tokenizer(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Forward
    print("\nTesting forward pass...")
    image_embeddings, text_embeddings = model(
        images,
        encoding['input_ids'],
        encoding['attention_mask']
    )

    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")

    similarity = model.get_similarity_matrix(image_embeddings, text_embeddings)
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity matrix:\n{similarity}")

    print("\nTesting phase configuration...")
    model.configure_phase(1, config)
    model.print_trainable_params()

    model.configure_phase(2, config)
    model.print_trainable_params()
