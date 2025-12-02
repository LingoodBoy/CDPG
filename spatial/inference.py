import os
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.clip_model import CLIPModel


class EmbeddingExtractor:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'mps' else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']

        self.model = CLIPModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully")

        image_size = self.config['model']['image_encoder']['image_size']
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['text_encoder']['backbone']
        )
        print(f"Tokenizer loaded for inference, vocab_size={len(self.tokenizer)}")
        self.text_max_length = self.config['model']['text_encoder']['max_length']

    @torch.no_grad()
    def extract_image_embedding(self, image_path: str) -> np.ndarray:

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        embedding = self.model.image_encoder(image_tensor)
        return embedding.cpu().numpy()[0]

    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> np.ndarray:

        encoding = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        embedding = self.model.text_encoder(input_ids, attention_mask)
        return embedding.cpu().numpy()[0]

    @torch.no_grad()
    def extract_batch_images(self, image_dir: str, output_path: str):

        image_paths = sorted(Path(image_dir).glob('*.png'))
        print(f"Found {len(image_paths)} images in {image_dir}")

        embeddings = []
        filenames = []

        for img_path in tqdm(image_paths, desc="Extracting image embeddings"):
            emb = self.extract_image_embedding(str(img_path))
            embeddings.append(emb)
            filenames.append(img_path.name)

        embeddings = np.array(embeddings)  # [N, 64]
        np.savez(
            output_path,
            embeddings=embeddings,
            filenames=filenames
        )
        print(f"Saved {len(embeddings)} embeddings to {output_path}")
        print(f"  Shape: {embeddings.shape}")

    @torch.no_grad()
    def extract_batch_texts(self, json_path: str, output_path: str):

        import json

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📂 Found {len(data)} texts in {json_path}")

        embeddings = []
        filenames = []
        texts = []

        for item in tqdm(data, desc="Extracting text embeddings"):
            text = item['text']

            emb = self.extract_text_embedding(text)
            embeddings.append(emb)
            filenames.append(item['image'])
            texts.append(text)

        embeddings = np.array(embeddings)  # [N, 64]
        np.savez(
            output_path,
            embeddings=embeddings,
            filenames=filenames,
            texts=texts
        )
        print(f"Saved {len(embeddings)} embeddings to {output_path}")
        print(f"   Shape: {embeddings.shape}")

    @torch.no_grad()
    def extract_all_cities(
        self,
        image_base_dir: str,
        text_base_dir: str,
        output_dir: str,
        cities: list = ['sh', 'nj', 'nc']
    ):

        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("Extracting embeddings for all cities (aligned by image filename)")
        print("="*70)

        for city in cities:
            print(f"\nProcessing city: {city.upper()}")

            image_dir = Path(image_base_dir) / city
            text_json = Path(text_base_dir) / f"{city}_results.json"

            if not image_dir.exists():
                print(f"Image directory not found: {image_dir}")
                continue

            if not text_json.exists():
                print(f"Text JSON not found: {text_json}")
                continue

            with open(text_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text_dict = {item['image']: item['text'] for item in data}
            print(f"Loaded {len(text_dict)} text entries from JSON")

            image_files = list(image_dir.glob('*.png'))

            def extract_number(path):
                try:
                    return int(path.stem)  # '1234.png' -> 1234
                except ValueError:
                    return float('inf') 

            image_files = sorted(image_files, key=extract_number)
            print(f"Found {len(image_files)} images")

            image_embeddings = []
            text_embeddings = []
            missing_texts = []

            for img_path in tqdm(image_files, desc=f"Extracting {city} embeddings"):

                img_emb = self.extract_image_embedding(str(img_path))
                image_embeddings.append(img_emb)

                img_name = img_path.name
                if img_name in text_dict:
                    text = text_dict[img_name]
                    txt_emb = self.extract_text_embedding(text)
                    text_embeddings.append(txt_emb)
                else:

                    missing_texts.append(img_name)
                    txt_emb = self.extract_text_embedding("")
                    text_embeddings.append(txt_emb)

            if missing_texts:
                print(f" Warning: {len(missing_texts)} images have no text in JSON")
                print(f"   First few: {missing_texts[:5]}")

            image_embeddings = np.array(image_embeddings)  # [N, 64]
            text_embeddings = np.array(text_embeddings)    # [N, 64]

            image_output = output_path / f"{city}_image_embeddings.npy"
            text_output = output_path / f"{city}_text_embeddings.npy"

            np.save(image_output, image_embeddings)
            np.save(text_output, text_embeddings)

            print(f"Saved to:")
            print(f"   {image_output} - shape {image_embeddings.shape}")
            print(f"   {text_output} - shape {text_embeddings.shape}")

        print("\n" + "="*70)
        print("All embeddings extracted successfully!")
        print(f"Output directory: {output_dir}")
        print("="*70)

        print("\nSummary:")
        for city in cities:
            image_file = output_path / f"{city}_image_embeddings.npz"
            text_file = output_path / f"{city}_text_embeddings.npz"

            if image_file.exists():
                data = np.load(image_file)
                print(f"   {city.upper()} images: {data['embeddings'].shape[0]} samples")

            if text_file.exists():
                data = np.load(text_file)
                print(f"   {city.upper()} texts:  {data['embeddings'].shape[0]} samples")

    def compute_similarity(self, image_path: str, text: str) -> float:

        image_emb = self.extract_image_embedding(image_path)
        text_emb = self.extract_text_embedding(text)

        similarity = np.dot(image_emb, text_emb)
        return float(similarity)

    def retrieve_top_k(
        self,
        query_image: str,
        candidate_texts: list,
        k: int = 5
    ) -> list:

        image_emb = self.extract_image_embedding(query_image)

        similarities = []
        for text in tqdm(candidate_texts, desc="Computing similarities"):
            text_emb = self.extract_text_embedding(text)
            sim = np.dot(image_emb, text_emb)
            similarities.append((text, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


def main():
    parser = argparse.ArgumentParser(description='embeddings')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint')
    parser.add_argument('--mode', type=str,
                        choices=['image', 'text', 'batch', 'similarity', 'all_cities'],
                        required=True, help='running mode')

    # Image/Text mode
    parser.add_argument('--image', type=str, help='image path')
    parser.add_argument('--text', type=str, help='text content')

    # Batch mode
    parser.add_argument('--image-dir', type=str, help='image path')
    parser.add_argument('--text-json', type=str, help='text json path')
    parser.add_argument('--output', type=str, help='output dir')

    # All cities mode
    parser.add_argument('--image-base-dir', type=str, help='image path')
    parser.add_argument('--text-base-dir', type=str, help='text json path')
    parser.add_argument('--output-dir', type=str, help='output dir')
    parser.add_argument('--cities', type=str, nargs='+', default=['sh', 'nj', 'nc'],
                        help='city list（default: sh nj nc）')

    parser.add_argument('--device', type=str, default='cuda', help='running device')

    args = parser.parse_args()

    extractor = EmbeddingExtractor(args.checkpoint, device=args.device)

    if args.mode == 'image':
        if not args.image:
            raise ValueError("--image is required for image mode")

        embedding = extractor.extract_image_embedding(args.image)
        print(f"\nImage Embedding:")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.4f}")
        print(f"   First 10 values: {embedding[:10]}")

    elif args.mode == 'text':
        if not args.text:
            raise ValueError("--text is required for text mode")

        embedding = extractor.extract_text_embedding(args.text)
        print(f"\nText Embedding:")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.4f}")
        print(f"   First 10 values: {embedding[:10]}")

    elif args.mode == 'batch':
        if not args.image_dir or not args.output:
            raise ValueError("--image-dir and --output are required for batch mode")

        extractor.extract_batch_images(args.image_dir, args.output)

    elif args.mode == 'similarity':
        if not args.image or not args.text:
            raise ValueError("--image and --text are required for similarity mode")

        similarity = extractor.compute_similarity(args.image, args.text)
        print(f"\n Similarity: {similarity:.4f}")
        if similarity > 0.7:
            print("    Very similar (high match)")
        elif similarity > 0.5:
            print("    Similar (moderate match)")
        elif similarity > 0.3:
            print("   Somewhat similar (weak match)")
        else:
            print("   Not similar (no match)")

    elif args.mode == 'all_cities':
        if not args.image_base_dir or not args.text_base_dir or not args.output_dir:
            raise ValueError("--image-base-dir, --text-base-dir, and --output-dir are required for all_cities mode")

        extractor.extract_all_cities(
            image_base_dir=args.image_base_dir,
            text_base_dir=args.text_base_dir,
            output_dir=args.output_dir,
            cities=args.cities
        )


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Example Usage:")
    print("="*70)
    print("\n1. Extract single image embedding:")
    print("   python inference.py --checkpoint checkpoints/best_model.pt --mode image --image path/to/image.png")
    print("\n2. Extract single text embedding:")
    print("   python inference.py --checkpoint checkpoints/best_model.pt --mode text --text 'A dense urban area'")
    print("\n3. Batch extract image embeddings:")
    print("   python inference.py --checkpoint checkpoints/best_model.pt --mode batch --image-dir data/satellite_images/sh --output sh_embeddings.npz")
    print("\n4. Compute image-text similarity:")
    print("   python inference.py --checkpoint checkpoints/best_model.pt --mode similarity --image path/to/image.png --text 'A dense urban area'")
    print("\n5. Extract ALL cities (sh/nj/nc) image and text embeddings (aligned by filename):")
    print("   python inference.py --checkpoint checkpoints/best_model.pt --mode all_cities \\")
    print("     --image-base-dir /path/to/satellite_images \\")
    print("     --text-base-dir /path/to/llm_texts \\")
    print("     --output-dir embeddings_output")
    print("\n   This will generate 6 .npy files (aligned by image filename 0.png, 1.png, ...):")
    print("     - sh_image_embeddings.npy [4505, 64], sh_text_embeddings.npy [4505, 64]")
    print("     - nj_image_embeddings.npy [8000, 64], nj_text_embeddings.npy [8000, 64]")
    print("     - nc_image_embeddings.npy [7702, 64], nc_text_embeddings.npy [7702, 64]")
    print("\n" + "="*70 + "\n")

    main()
