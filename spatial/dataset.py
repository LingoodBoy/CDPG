import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


class SatelliteImageTextDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        text_dir: str,
        cities: List[str],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        cross_city_test: bool = False,
        image_size: int = 256,
        text_max_length: int = 128,
        tokenizer_name: str = "bert-base-uncased",
        seed: int = 42
    ):
        self.image_dir = Path(image_dir)
        self.text_dir = Path(text_dir)
        self.split = split
        self.image_size = image_size
        self.text_max_length = text_max_length
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)
            print(f"Tokenizer loaded: {tokenizer_name}, vocab_size={len(self.tokenizer)}")
        except (OSError, ValueError, Exception) as e:
            print(f"Cannot load pretrained tokenizer, creating basic tokenizer")
            from transformers import BertTokenizer, BasicTokenizer, WordpieceTokenizer

            class SimpleTokenizer:
                def __init__(self, max_length=128):
                    self.max_length = max_length

                    self.vocab = {chr(i): i for i in range(32, 127)}  # ASCII
                    self.vocab['[PAD]'] = 0
                    self.vocab['[UNK]'] = 1
                    self.vocab['[CLS]'] = 2
                    self.vocab['[SEP]'] = 3
                    self.vocab['[MASK]'] = 4

                def __call__(self, texts, max_length=None, padding='max_length',
                           truncation=True, return_tensors='pt'):
                    if isinstance(texts, str):
                        texts = [texts]

                    max_len = max_length or self.max_length
                    input_ids_list = []
                    attention_mask_list = []

                    for text in texts:

                        tokens = text.lower().split()[:max_len-2]

                        token_ids = [2]  # [CLS]
                        for token in tokens:
                            for char in token[:10]:
                                token_ids.append(self.vocab.get(char, 1))
                        token_ids.append(3) 

                        attention_mask = [1] * len(token_ids)
                        if len(token_ids) < max_len:
                            padding_length = max_len - len(token_ids)
                            token_ids.extend([0] * padding_length)
                            attention_mask.extend([0] * padding_length)
                        else:
                            token_ids = token_ids[:max_len]
                            attention_mask = attention_mask[:max_len]

                        input_ids_list.append(token_ids)
                        attention_mask_list.append(attention_mask)

                    if return_tensors == 'pt':
                        import torch
                        return {
                            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
                        }
                    return {'input_ids': input_ids_list, 'attention_mask': attention_mask_list}

            self.tokenizer = SimpleTokenizer(max_length=text_max_length)
            print(f"Using simple character-level tokenizer (offline mode)")

        if split == "train":
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.samples = self._load_samples(
            cities,
            split,
            train_ratio,
            val_ratio,
            test_ratio,
            cross_city_test,
            seed
        )

        print(f"{split.upper()} set loaded: {len(self.samples)} samples")

    def _load_samples(
        self,
        cities: List[str],
        split: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        cross_city_test: bool,
        seed: int
    ) -> List[Tuple[str, str, str]]:
        all_samples = []

        for city in cities:

            json_path = self.text_dir / f"{city}_results.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            city_samples = []
            for item in data:
                image_name = item['image']
                text = item['text']

                image_path = self.image_dir / city / image_name

                if not image_path.exists():
                    print(f" Warning: {image_path} not found, skipping")
                    continue

                city_samples.append((city, str(image_path), text))

            all_samples.extend(city_samples)
            print(f"📂 Loaded {len(city_samples)} samples from {city}")

        if cross_city_test:
            if split == "test":
                samples = [s for s in all_samples if s[0] == "nc"]
            else:
                train_val_samples = [s for s in all_samples if s[0] in ["sh", "nj"]]
                samples = self._split_train_val(train_val_samples, split, train_ratio, val_ratio, seed)
        else:
            samples = self._split_train_val_test(all_samples, split, train_ratio, val_ratio, test_ratio, seed)

        return samples

    def _split_train_val(
        self,
        samples: List[Tuple[str, str, str]],
        split: str,
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> List[Tuple[str, str, str]]:
        import random
        random.seed(seed)
        random.shuffle(samples)

        train_size = int(len(samples) * train_ratio / (train_ratio + val_ratio))

        if split == "train":
            return samples[:train_size]
        else:  # val
            return samples[train_size:]

    def _split_train_val_test(
        self,
        samples: List[Tuple[str, str, str]],
        split: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> List[Tuple[str, str, str]]:
        import random
        random.seed(seed)
        random.shuffle(samples)

        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        if split == "train":
            return samples[:train_size]
        elif split == "val":
            return samples[train_size:train_size + val_size]
        else:  # test
            return samples[train_size + val_size:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'image': Tensor [3, H, W],
                'input_ids': Tensor [max_length],
                'attention_mask': Tensor [max_length],
                'city': str,
                'image_path': str
            }
        """
        city, image_path, text = self.samples[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        encoding = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'city': city,
            'image_path': image_path
        }


def create_dataloaders(
    config: dict,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_config = config['data']
    model_config = config['model']

    train_dataset = SatelliteImageTextDataset(
        image_dir=data_config['image_dir'],
        text_dir=data_config['text_dir'],
        cities=data_config['cities'],
        split='train',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        cross_city_test=data_config['cross_city_test'],
        image_size=model_config['image_encoder']['image_size'],
        text_max_length=model_config['text_encoder']['max_length'],
        tokenizer_name=model_config['text_encoder']['backbone'],
        seed=config['training']['seed']
    )

    val_dataset = SatelliteImageTextDataset(
        image_dir=data_config['image_dir'],
        text_dir=data_config['text_dir'],
        cities=data_config['cities'],
        split='val',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        cross_city_test=data_config['cross_city_test'],
        image_size=model_config['image_encoder']['image_size'],
        text_max_length=model_config['text_encoder']['max_length'],
        tokenizer_name=model_config['text_encoder']['backbone'],
        seed=config['training']['seed']
    )

    test_dataset = SatelliteImageTextDataset(
        image_dir=data_config['image_dir'],
        text_dir=data_config['text_dir'],
        cities=data_config['cities'],
        split='test',
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        cross_city_test=data_config['cross_city_test'],
        image_size=model_config['image_encoder']['image_size'],
        text_max_length=model_config['text_encoder']['max_length'],
        tokenizer_name=model_config['text_encoder']['backbone'],
        seed=config['training']['seed']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=4,
        num_workers=0
    )

    print("\nTesting dataloader...")
    batch = next(iter(train_loader))

    print(f"Image shape: {batch['image'].shape}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Cities: {batch['city']}")
    print(f"First text (decoded): {train_loader.dataset.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
    print("\nDataloader test passed!")
