"""Config-driven training scaffold for AMPlify-v2 experiments.

This is a minimal, readable scaffold meant to be extended. It loads YAML config,
instantiates backbone and pooling modules, and runs a simple training loop skeleton.
"""
import argparse
import datetime
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from . import backbone as backbone_module
from . import pooling as pooling_module
from .model import AMPlifyV2


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}


def seq_to_onehot(seq: str, max_len: int) -> np.ndarray:
    arr = np.zeros((max_len, len(AA_LIST)), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        idx = AA_TO_IDX.get(aa, None)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def read_fasta(path: str, limit: int = None) -> List[Tuple[str, str]]:
    records = []
    with open(path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    if limit:
        return records[:limit]
    return records


class SimpleSeqDataset(torch.utils.data.Dataset):
    def __init__(self, pos_fasta: str, neg_fasta: str, max_len: int = 200, limit_per_class: int = 1000):
        pos = read_fasta(pos_fasta, limit_per_class)
        neg = read_fasta(neg_fasta, limit_per_class)
        self.seqs = [s for _, s in pos] + [s for _, s in neg]
        self.labels = [1] * len(pos) + [0] * len(neg)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        x = seq_to_onehot(seq, self.max_len)
        return x, self.labels[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = np.stack(xs)
    return torch.from_numpy(xs), torch.tensor(ys, dtype=torch.float32)


def get_git_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return out
    except Exception:
        return "unknown"


def build_components(cfg: dict) -> AMPlifyV2:
    bcfg = cfg.get("model", {})
    backbone_type = bcfg.get("backbone", "lstm")
    pooling_type = bcfg.get("pooling", "context")
    input_dim = bcfg.get("input_dim", 20)
    d_model = bcfg.get("d_model", 256)

    if backbone_type == "lstm":
        backbone = backbone_module.BiLSTMBackbone(input_dim=input_dim, hidden_dim=d_model // 2)
    elif backbone_type == "transformer":
        backbone = backbone_module.TransformerBackbone(input_dim=input_dim, d_model=d_model, num_layers=bcfg.get("num_layers", 2))
    elif backbone_type == "conformer":
        backbone = backbone_module.ConformerBackbone(input_dim=input_dim, d_model=d_model, num_blocks=bcfg.get("num_blocks", 2))
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    if pooling_type in {"context", "singlehead"}:
        pooling = pooling_module.SingleHeadAttentionPooling(d_model)
    elif pooling_type == "multihead":
        pooling = pooling_module.MultiHeadPooling(d_model, num_heads=bcfg.get("pooling_heads", 4))
    elif pooling_type == "cls":
        pooling = pooling_module.CLSPooling(d_model)
    else:
        raise ValueError(f"Unknown pooling: {pooling_type}")

    model = AMPlifyV2(backbone=backbone, pooling=pooling)
    # infer pooled dim by running a dummy tensor
    device = torch.device("cpu")
    dummy_x = torch.randn(1, 16, bcfg.get("input_dim", 20))
    with torch.no_grad():
        h = backbone(dummy_x)
        pooled, _ = pooling(h)
    pooled_dim = pooled.shape[-1]
    # no global or esm dims here; they'll be added later if used
    model.build_classifier(pooled_dim=pooled_dim, output_dim=1, global_feat_dim=None, esm_embed_dim=None)
    return model


def main(argv: Any = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="amplify/experiments/amplify_v2/config.yaml")
    parser.add_argument("--model", default="amplify_v2")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    model = build_components(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr = float(cfg.get("training", {}).get("lr", 1e-3))
    criterion = nn.BCEWithLogitsLoss()

    save_dir = Path(cfg.get("logging", {}).get("save_dir", "amplify/experiments/logs"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log run metadata
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_log = {
        "run_id": run_id,
        "git": get_git_hash(),
        "config": cfg,
    }
    (save_dir / f"run_{run_id}_meta.json").write_text(json.dumps(run_log, indent=2))

    # Prepare dataset (very small quick-run example)
    data_dir = Path("data")
    pos_train = data_dir / "AMPlify_AMP_train_common.fa"
    neg_train = data_dir / "AMPlify_non_AMP_train_balanced.fa"
    pos_test = data_dir / "AMPlify_AMP_test_common.fa"
    neg_test = data_dir / "AMPlify_non_AMP_test_balanced.fa"

    if not (pos_train.exists() and neg_train.exists() and pos_test.exists() and neg_test.exists()):
        print("Data files not found under data/. Skipping quick training run.")
        return

    max_len = int(cfg.get("model", {}).get("max_len", 200))
    limit_per_class = int(cfg.get("training", {}).get("limit_per_class", 500))
    train_ds = SimpleSeqDataset(str(pos_train), str(neg_train), max_len=max_len, limit_per_class=limit_per_class)
    test_ds = SimpleSeqDataset(str(pos_test), str(neg_test), max_len=max_len, limit_per_class=limit_per_class)

    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optional ESM support: try to import real ESM, otherwise provide a learned embedding fallback
    use_esm = cfg.get("optional", {}).get("use_esm", False)
    esm_dim = cfg.get("optional", {}).get("esm_dim", None)
    esm_available = False
    esm_model = None
    aa_embedding = None
    if use_esm:
        try:
            import esm

            esm_available = True
            print("ESM package available — using ESM for embeddings (if configured).")
            # Note: loading large pretrained weights may be slow / require internet; user can adjust config
        except Exception:
            print("ESM package not available; falling back to learned amino-acid embeddings.")
            esm_available = False
            if esm_dim is None:
                esm_dim = 128
            aa_embedding = nn.Linear(len(AA_LIST), esm_dim)
            aa_embedding.to(device)

    # If ESM fallback or not using ESM, ensure model classifier accounts for esm dim
    # Rebuild classifier if needed to include esm_dim
    if use_esm and not esm_available:
        # infer pooled dim
        with torch.no_grad():
            dummy_x = torch.randn(1, 16, cfg.get("model", {}).get("input_dim", 20)).to(device)
            h = model.backbone(dummy_x)
            pooled, _ = model.pooling(h)
        pooled_dim = pooled.shape[-1]
        model.build_classifier(pooled_dim=pooled_dim, output_dim=1, global_feat_dim=None, esm_embed_dim=esm_dim)

    # Build optimizer including aa_embedding parameters if present
    if aa_embedding is not None:
        optimizer = optim.Adam(list(model.parameters()) + list(aa_embedding.parameters()), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = int(cfg.get("training", {}).get("epochs", 1))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # xb shape: (batch, seq_len, aa_dim)
            # If model expects one-hot inputs, pass directly; if using ESM, compute esm_emb and pass pooled
            optimizer.zero_grad()
            if use_esm and esm_available:
                # Not implementing full ESM extraction here — placeholder
                esm_emb = None
                out, _ = model(xb, global_feats=None, esm_emb=esm_emb)
            elif use_esm and (not esm_available):
                # compute learned aa embedding by summing per-position learned vectors
                b, seq_len, aa_dim = xb.shape
                xb_flat = xb.view(-1, aa_dim).to(device)
                emb_flat = aa_embedding(xb_flat)
                emb = emb_flat.view(b, seq_len, -1)
                # average pool to get per-sequence embedding
                esm_emb = emb.mean(dim=1)
                out, _ = model(xb.to(torch.float32), esm_emb=esm_emb)
            else:
                out, _ = model(xb.to(torch.float32))
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    ys_true = []
    ys_score = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if use_esm and (not esm_available):
                b, seq_len, aa_dim = xb.shape
                xb_flat = xb.view(-1, aa_dim).to(device)
                emb_flat = aa_embedding(xb_flat)
                emb = emb_flat.view(b, seq_len, -1)
                esm_emb = emb.mean(dim=1)
                out, _ = model(xb, esm_emb=esm_emb)
            else:
                out, _ = model(xb)
            probs = torch.sigmoid(out)
            ys_true.extend(yb.cpu().numpy().tolist())
            ys_score.extend(probs.cpu().numpy().tolist())

    import numpy as _np
    from . import metrics

    roc = metrics.roc_auc(_np.array(ys_true), _np.array(ys_score))
    pr = metrics.pr_auc(_np.array(ys_true), _np.array(ys_score))
    print(f"Test ROC AUC: {roc:.4f} — PR AUC: {pr:.4f}")

    # Save metrics
    (save_dir / f"run_{run_id}_metrics.json").write_text(json.dumps({"roc_auc": roc, "pr_auc": pr}, indent=2))

    # Save final model checkpoint
    ckpt_path = save_dir / f"{args.model}_final_{run_id}.pt"
    save_dict = {"model_state_dict": model.state_dict(), "config": cfg}
    if aa_embedding is not None:
        save_dict["aa_embedding_state_dict"] = aa_embedding.state_dict()
    torch.save(save_dict, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
