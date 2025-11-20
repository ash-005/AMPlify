"""Run quick debugging checks: labels, logits, loss usage, gradients, embeddings.

This script executes the quick checks recommended in the debugging plan.
Run it with the same config used for training (defaults to config_esm.yaml).
"""
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from . import train as train_mod


def print_label_distribution(ds, name="dataset"):
    labels = np.array(ds.labels)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution ({name}):", dict(zip(unique, counts)))


def run_checks(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    print("Loaded config:", cfg_path)

    # Build model and data like in train.py
    model = train_mod.build_components(cfg)
    device = torch.device("cpu")
    model.to(device)

    data_dir = Path("data")
    pos_train = data_dir / "AMPlify_AMP_train_common.fa"
    neg_train = data_dir / "AMPlify_non_AMP_train_balanced.fa"
    pos_test = data_dir / "AMPlify_AMP_test_common.fa"
    neg_test = data_dir / "AMPlify_non_AMP_test_balanced.fa"

    if not (pos_train.exists() and neg_train.exists() and pos_test.exists() and neg_test.exists()):
        print("Data files missing under data/. Cannot run checks.")
        return

    max_len = int(cfg.get("model", {}).get("max_len", 200))
    limit_per_class = int(cfg.get("training", {}).get("limit_per_class", 500))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))

    train_ds = train_mod.SimpleSeqDataset(str(pos_train), str(neg_train), max_len=max_len, limit_per_class=limit_per_class)
    test_ds = train_mod.SimpleSeqDataset(str(pos_test), str(neg_test), max_len=max_len, limit_per_class=limit_per_class)

    print_label_distribution(train_ds, "train")
    print_label_distribution(test_ds, "test")

    # Print some sample sequences/labels
    print("First 5 train samples (seq_len, label):")
    for i in range(min(5, len(train_ds))):
        s, l = train_ds.seqs[i], train_ds.labels[i]
        print(i, len(s), l, s[:60])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_mod.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=train_mod.collate_fn)

    # Quick logits/prob check
    xb, yb = next(iter(train_loader))
    xb = xb.to(device)
    yb = yb.to(device)

    model.eval()
    with torch.no_grad():
        logits, attn = model(xb.float())
        probs = torch.sigmoid(logits)

    print("Batch shapes: xb", xb.shape, "yb", yb.shape)
    print("logits stats:", float(logits.mean()), float(logits.std()), float(logits.min()), float(logits.max()))
    print("probs stats:", float(probs.mean()), float(probs.std()), float(probs.min()), float(probs.max()))
    print("sample logits:", logits[:10].cpu().numpy())
    print("sample probs:", probs[:10].cpu().numpy())

    # Loss usage check
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, yb.float())
    print("Loss computed with BCEWithLogitsLoss():", float(loss))

    # Gradient check: do one backward and inspect grads
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    logits_train, _ = model(xb.float())
    loss_train = loss_fn(logits_train, yb.float())
    loss_train.backward()

    none_grad = []
    zero_grad = []
    tiny_grad = []
    some_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grad.append(name)
        else:
            gnorm = p.grad.data.norm().item()
            if gnorm == 0.0:
                zero_grad.append((name, gnorm))
            elif gnorm < 1e-7:
                tiny_grad.append((name, gnorm))
            else:
                some_grad.append((name, gnorm))

    print("Parameters with no grad:", none_grad[:10])
    print("Parameters with zero grad (sample):", zero_grad[:10])
    print("Parameters with tiny grad (sample):", tiny_grad[:10])
    print("Parameters with non-trivial grad (sample):", some_grad[:10])

    # Inspect first-layer weights / pooling context
    print("--- first-layer weight norms ---")
    b = model.backbone
    if hasattr(b, "rnn"):
        for n, p in b.rnn.named_parameters():
            print(f"backbone.rnn.{n} norm:", float(p.data.norm()))
    if hasattr(b, "input_proj"):
        print("backbone.input_proj weight norm:", float(b.input_proj.weight.data.norm()))
    pool = model.pooling
    if hasattr(pool, "context"):
        print("pool.context weight norm:", float(pool.context.weight.data.norm()))
    if hasattr(pool, "attn"):
        print("multihead attn head 0 weight norm:", float(pool.attn[0].weight.data.norm()))

    print("Done checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="amplify/experiments/amplify_v2/config_esm.yaml")
    args = parser.parse_args()
    run_checks(args.config)
