
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, BertModel, BertConfig


class TextEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        embedding_dim: int = 64,
        dropout: float = 0.1,
        freeze: bool = True,
        local_files_only: bool = False 
    ):

        super().__init__()

        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        try:

            self.bert = AutoModel.from_pretrained(backbone, local_files_only=False)
            bert_config = AutoConfig.from_pretrained(backbone, local_files_only=False)
            bert_output_dim = bert_config.hidden_size
            print(f"BERT backbone loaded from pretrained: {backbone}, output_dim={bert_output_dim}")
        except (OSError, ValueError) as e:

            print(f"Cannot load pretrained BERT (offline mode or no cache)")
            print(f"   Using randomly initialized BERT-base architecture")

            bert_config = BertConfig(
                vocab_size=30522,  # BERT
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512
            )
            self.bert = BertModel(bert_config)
            bert_output_dim = bert_config.hidden_size

        # Projection Head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
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
        for param in self.bert.parameters():
            param.requires_grad = False
        print("Text Encoder backbone frozen")

    def unfreeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        print("Text Encoder backbone unfrozen")

    def unfreeze_last_n_layers(self, n: int = 3):
        total_layers = len(self.bert.encoder.layer)

        if n > total_layers:
            print(f"Warning: n={n} > total_layers={total_layers}, unfreezing all layers")
            n = total_layers

        for layer in self.bert.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.bert.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Text Encoder: unfroze last {n}/{total_layers} layers")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_output_dim]

        embedding = self.projection(cls_output)  # [batch_size, embedding_dim]

        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


if __name__ == '__main__':
    from transformers import AutoTokenizer

    model = TextEncoder(
        backbone="bert-base-uncased",
        embedding_dim=64,
        freeze=True
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = [
        "This image shows a dense mixed-use urban area.",
        "A grid-like road network with a major highway provides connectivity.",
        "Green spaces and parks are scattered throughout the development.",
        "Industrial zones with large buildings and warehouses."
    ]

    # Tokenize
    encoding = tokenizer(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Forward
    embedding = model(encoding['input_ids'], encoding['attention_mask'])
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding[0]).item():.4f} (should be ~1.0)")

    print(f"\nTrainable params (frozen): {model.get_trainable_params():,}")

    model.unfreeze_last_n_layers(3)
    print(f"Trainable params (last 3 layers unfrozen): {model.get_trainable_params():,}")

    model.unfreeze_backbone()
    print(f"Trainable params (all unfrozen): {model.get_trainable_params():,}")

    print("\n Text Encoder test passed!")
