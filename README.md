# CDPG: Conditional Diffusion-Based Neural Parameter Generation for Few-Shot Cross-City Mobile Traffic Prediction

## Project Structure

```
CDPG/
├── Pretrain/          # Mobile traffic prediction model pretraining and few-shot fine-tuning
├── parameter_generate/               # Diffusion-based Neural Parameter Generation
├── spatial/               # LLM-Enhanced Spatial Context Extraction
├── temporal/               # Pattern-Guided Temporal Context Extraction
├── data/              # Data directory
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n cdpg python=3.8
conda activate cdpg
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### 2. Data Preparation

Place data files in the following structure:
```
data/
├── handled/           # Processed traffic data
├── model_param/       # Pre-trained model parameters
├── spatial_llm_embedding/  # Spatial embeddings
└── temporal_embedding/     # Temporal embeddings
```

### 3. Run Pipeline

#### Step 1: Pretrain Traffic Prediction Model
```bash
cd ./predict
python main_parallel.py --mode pretrain --ifnewname 2 --aftername traffic_pretrain --epochs 50 --message_dim 1 --test_data nc --batch_size 64 --test_batch_size 32
```

#### Step 2: Spatial Context Extraction
```bash
cd ./spatial
python train_simple.py --config config_simple.yaml

python inference.py --checkpoint checkpoints/best_model.pt --mode all_cities --image-base-dir data/satellite_images --text-base-dir data/llm_texts
--output-dir embeddings_output
```

#### Step 3: Temporal Context Extraction
```bash
cd ./temporal
python ensemble_clustering_temporal_v2.py --exp 2
```

#### Step 4: Generate Parameters via Diffusion
```bash
cd ./parameter_generate
python main.py --expIndex 101 --targetDataset nc --modeldim 512 --epochs 80000 --diffusionstep 500 --batchsize 128 --modelparam data/model_param/ModelParams.npy --spatial data/spatial_llm_embedding/spatial_llm_all_emb.npy --temporal data/temporal_embedding/temporal_emb_exp2_target_nc.npy
```

#### Step 5: Finetune on Target City
```bash
cd ./predict
python main_parallel.py --mode finetuning --test_data nc --epochs 50 --target_days 3 --batch_size 10 --test_batch_size 8 --ifnewname 1 --aftername finetune_3days --params_file parameter_generate/Output/exp101/sampleSeq_RealParams_101.npy
```

## License

MIT License

