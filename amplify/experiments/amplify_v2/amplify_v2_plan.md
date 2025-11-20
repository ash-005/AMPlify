# AMPlify-v2 Refactor and Experimentation Plan

## Goals
- Add modular neural backbones (BiLSTM, Transformer, Conformer)
- Add improved pooling layers (multi-head pooling, CLS pooling)
- Optional support for ESM embeddings
- Config-driven experiments
- Clean training/evaluation pipeline for ablation studies

## Tasks for Copilot

### 1. Project Structure
- Create `amplify/experiments/amplify_v2/`
- Add `backbone.py`, `pooling.py`, `model.py`, `config.yaml`, `train.py`, `README.md`

### 2. Backbone Modules
- Implement `BiLSTMBackbone` (reuse existing code where possible)
- Implement `TransformerBackbone` (2–4 layers)
- Implement `ConformerBackbone` (lightweight)

### 3. Pooling Modules
- Port original context attention pooling
- Add `MultiHeadPooling`
- Add `CLSPooling`

### 4. Model Wrapper
- Create `AMPlifyV2` that accepts `{backbone, pooling, global_features, esm_embeddings}`
- Ensure forward pass handles variable sequence lengths

### 5. Config System
- Add `config.yaml` entries:
  - backbone: "lstm" | "transformer" | "conformer"
  - pooling: "context" | "multihead" | "cls"
  - optimizer, lr, dropout, etc.
- Update `train.py` to load config and instantiate modules dynamically

### 6. Evaluation Tools
- Add metrics: ROC AUC, PR AUC, confusion matrix
- Add attention visualization utilities

### 7. Logging
- Add experiment log folder
- Save config, metrics, and model state dict per run
- Include git commit hash for reproducibility

### 8. Optional Enhancements
- Add ESM2 embedding support with a flag in config
- Add global physico-chemical feature extraction module
