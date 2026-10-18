#!/usr/bin/env python3
"""Create predicted-only parent class files from the MGSFormer NoParent model."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from parent_context_mgsformer import (
    GroupBundleDataset,
    MGSFormerOracleParentsClassifier,
    StandardScaler4D,
    load_group_samples,
    set_seed,
)


def save_predictions(model, samples, dataset, split, out_dir, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred = []
    model.eval()
    with torch.no_grad():
        for xb, parent_b, _ in loader:
            logits = model(xb.to(device), parent_b.to(device))
            pred.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int64))
    pred = np.concatenate(pred, axis=0)
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        np.save(split_dir / f"{sample.group_id}.npy", pred[i])
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parent_cap", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    samples = {}
    for split in ("train", "val", "test"):
        samples[split] = load_group_samples(
            args.data_dir, split, args.num_classes, args.parent_cap, "none"
        )
    train_x = np.stack([s.x for s in samples["train"]], axis=0).astype(np.float32)
    scaler = StandardScaler4D().fit(train_x)
    datasets = {k: GroupBundleDataset(v, scaler) for k, v in samples.items()}
    example_x, example_parent, _ = datasets["train"][0]
    model = MGSFormerOracleParentsClassifier(
        seq_len=int(example_x.shape[0]),
        node_num=int(example_x.shape[1]),
        feature_dim=int(example_x.shape[2]),
        parent_dim=int(example_parent.shape[1]),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_class=args.num_classes,
        use_parent_features=False,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        pred = save_predictions(model, samples[split], datasets[split], split, out, device, args.batch_size)
        counts[split] = {"groups": int(pred.shape[0]), "cities": int(pred.shape[1])}
    with open(out / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump({"source_checkpoint": args.checkpoint, "source_mode": "none", "seed": args.seed, "counts": counts}, stream, indent=2)


if __name__ == "__main__":
    main()
