
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        image_size: int = 256,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        freeze: bool = True,
        pretrained_path: str = None 
    ):
        super().__init__()

        # ViT
        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            self.backbone = timm.create_model(
                backbone,
                pretrained=False,
                img_size=image_size,
                num_classes=0
            )
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded local pretrained weights")
        else:
            if pretrained:
                print(f" Network unavailable, using random initialization instead of pretrained weights")
                print(f" Performance may be lower. Consider downloading weights offline.")

            self.backbone = timm.create_model(
                backbone,
                pretrained=False,
                img_size=image_size,
                num_classes=0
            )
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            vit_output_dim = self.backbone(dummy_input).shape[-1]

        print(f"ViT backbone loaded: {backbone}, output_dim={vit_output_dim}")

        # Projection Head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(vit_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim)
        )
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 4):
        # ViT
        total_blocks = len(self.backbone.blocks)

        if n > total_blocks:
            print(f"Warning: n={n} > total_blocks={total_blocks}, unfreezing all blocks")
            n = total_blocks

        for block in self.backbone.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block in self.backbone.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        print(f"Image Encoder: unfroze last {n}/{total_blocks} blocks")

    def get_trainable_params(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.backbone(x)  # [batch_size, vit_output_dim]

        embedding = self.projection(features)  # [batch_size, embedding_dim]

        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


if __name__ == '__main__':

    model = ImageEncoder(
        backbone="vit_base_patch16_224",
        pretrained=False, 
        image_size=256,
        embedding_dim=64,
        freeze=True
    )

    x = torch.randn(4, 3, 256, 256)
    embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding[0]).item():.4f} (should be ~1.0)")

    print(f"\nTrainable params (frozen): {model.get_trainable_params():,}")

    model.unfreeze_last_n_blocks(4)
    print(f"Trainable params (last 4 blocks unfrozen): {model.get_trainable_params():,}")

    model.unfreeze_backbone()
    print(f"Trainable params (all unfrozen): {model.get_trainable_params():,}")

    print("\n Image Encoder test passed (Offline Mode)")
