# Data Directory

## Required Data Files

### 1. Traffic Data (`handled/`)

Processed traffic data for three cities:
- `sh/`: Shanghai traffic data
- `nc/`: Nanchang traffic data
- `nj/`: Nanjing traffic data

Each city folder should contain:
- Adjacency matrix (`.npy`)
- Traffic flow data (`.npy`)

### 2. Model Parameters (`model_param/`)

Pre-trained model parameters:
- `ModelParams_v_Trans_all_6640.npy`: Combined parameters from source cities

### 3. Spatial Embeddings (`spatial_llm_embedding/`)

LLM-generated spatial embeddings:
- `spatial_llm_all_emb.npy`: Shape (20207, 128)
  - sh: [0:4505]
  - nc: [4505:12207]
  - nj: [12207:20207]

### 4. Temporal Embeddings (`temporal_embedding/`)

Ensemble clustering temporal embeddings:
- `temporal_emb_exp1_target_nj.npy`: Experiment 1 (target: nj)
- `temporal_emb_exp2_target_nc.npy`: Experiment 2 (target: nc)
- `temporal_emb_exp3_target_sh.npy`: Experiment 3 (target: sh)

All files shape: (20207, 128)

## Data Download

Due to data confidentiality concerns, data files are not included in this repository.