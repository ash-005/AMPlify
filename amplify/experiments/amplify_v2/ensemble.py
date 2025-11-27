"""Ensemble inference — load multiple trained models and aggregate predictions."""
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import yaml

from . import train as train_mod


def find_latest_checkpoints(log_dir: str, pattern: str = "*_final_*.pt", limit: int = 5) -> List[Path]:
    """Find the latest N checkpoint files."""
    log_path = Path(log_dir)
    checkpoints = sorted(log_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[:limit]


def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    
    # Build model using config from checkpoint
    model = train_mod.build_components(cfg)
    
    # If classifier needs to be rebuilt for ESM, do it here
    use_esm = cfg.get("optional", {}).get("use_esm", False)
    if use_esm:
        esm_dim = cfg.get("optional", {}).get("esm_dim", 128)
        # Rebuild classifier with ESM dim
        with torch.no_grad():
            dummy_x = torch.randn(1, 16, cfg.get("model", {}).get("input_dim", 20)).to(device)
            h = model.backbone(dummy_x)
            pooled, _ = model.pooling(h)
        pooled_dim = pooled.shape[-1]
        model.build_classifier(pooled_dim=pooled_dim, output_dim=1, global_feat_dim=None, esm_embed_dim=esm_dim)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load aa_embedding if present
    aa_embedding = None
    if "aa_embedding_state_dict" in ckpt:
        esm_dim = cfg.get("optional", {}).get("esm_dim", 128)
        aa_embedding = torch.nn.Linear(20, esm_dim)
        aa_embedding.load_state_dict(ckpt["aa_embedding_state_dict"])
        aa_embedding.to(device)
        aa_embedding.eval()
    
    return model, cfg, aa_embedding


def ensemble_predict(models: List[Tuple], data_loader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Run ensemble inference — average predictions from all models."""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            batch_probs = []
            for model, cfg, aa_embedding in models:
                use_esm = cfg.get("optional", {}).get("use_esm", False)
                
                if use_esm and aa_embedding is not None:
                    b, seq_len, aa_dim = xb.shape
                    xb_flat = xb.view(-1, aa_dim).to(device)
                    emb_flat = aa_embedding(xb_flat)
                    emb = emb_flat.view(b, seq_len, -1)
                    esm_emb = emb.mean(dim=1)
                    logits, _ = model(xb, esm_emb=esm_emb)
                else:
                    logits, _ = model(xb)
                
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())
            
            # Average across models
            ensemble_probs = np.mean(batch_probs, axis=0)
            all_probs.extend(ensemble_probs.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    
    return np.array(all_labels), np.array(all_probs)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="amplify/experiments/logs")
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--config", default="amplify/experiments/amplify_v2/config_tune_v1.yaml")
    args = parser.parse_args(argv)
    
    # Load test data
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = Path("data")
    pos_test = data_dir / "AMPlify_AMP_test_common.fa"
    neg_test = data_dir / "AMPlify_non_AMP_test_balanced.fa"
    
    max_len = int(cfg.get("model", {}).get("max_len", 200))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    
    test_ds = train_mod.SimpleSeqDataset(str(pos_test), str(neg_test), max_len=max_len, limit_per_class=None)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=train_mod.collate_fn)
    
    # Find and load latest checkpoints
    checkpoints = find_latest_checkpoints(args.log_dir, limit=args.num_models)
    print(f"Found {len(checkpoints)} checkpoints, loading {args.num_models}...")
    
    models = []
    for ckpt_path in checkpoints[:args.num_models]:
        model, cfg_loaded, aa_embedding = load_checkpoint(str(ckpt_path), device)
        models.append((model, cfg_loaded, aa_embedding))
        print(f"  Loaded: {ckpt_path.name}")
    
    # Run ensemble inference
    print(f"\nRunning ensemble prediction on {len(test_ds)} test samples...")
    ys_true, ys_score = ensemble_predict(models, test_loader, device)
    
    # Compute metrics
    from . import metrics
    
    roc = metrics.roc_auc(ys_true, ys_score)
    pr = metrics.pr_auc(ys_true, ys_score)
    
    print(f"\nEnsemble Results ({args.num_models} models):")
    print(f"  ROC AUC: {roc:.4f}")
    print(f"  PR AUC:  {pr:.4f}")
    
    # Save ensemble results
    log_path = Path(args.log_dir)
    result_file = log_path / f"ensemble_{args.num_models}_models_roc_{roc:.4f}.json"
    result_file.write_text(json.dumps({
        "num_models": args.num_models,
        "roc_auc": roc,
        "pr_auc": pr,
        "checkpoints": [str(c) for c in checkpoints[:args.num_models]],
    }, indent=2))
    print(f"\nSaved ensemble results to: {result_file}")


if __name__ == "__main__":
    main()
