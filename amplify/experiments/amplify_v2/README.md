# AMPlify v2 experiments

This folder contains scaffolding for AMPlify-v2 experiments: modular backbones, pooling layers, a model wrapper, config-driven training, metrics, and logging utilities.

Files:
- `backbone.py` — modular backbone implementations (BiLSTM / Transformer / Conformer)
- `pooling.py` — pooling layers (single-head attention, multi-head, CLS)
- `model.py` — `AMPlifyV2` composition model
- `train.py` — config-driven training scaffold
- `metrics.py` — evaluation utilities
- `config.yaml` — example config

Follow the `amplify_v2_plan.md` checklist to iterate with Copilot.
