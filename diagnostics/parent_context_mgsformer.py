import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


LABEL_NAMES = [
    "Excellent(0-35)",
    "Good(35-75)",
    "LightlyPolluted(75-115)",
    "ModeratelyPolluted(115-150)",
    "HeavilyPolluted(150-250)",
    "SeverelyPolluted(>250)",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, default=json_default)


def load_group_rows(data_dir: str, split: str) -> List[Dict[str, str]]:
    index_path = os.path.join(data_dir, "assets", split, "group_index.jsonl")
    with open(index_path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_parent_block(values: np.ndarray, dag_adj: np.ndarray, num_classes: int, parent_cap: int, mode: str) -> np.ndarray:
    """Encode oracle/predicted classes or lag-1 observed PM2.5 by parent slot."""
    node_num = int(values.shape[0])
    width = parent_cap * 2 if mode == "lag1" else parent_cap + parent_cap * num_classes
    block = np.zeros((node_num, width), dtype=np.float32)
    for dst in range(node_num):
        parents = np.flatnonzero(dag_adj[dst] > 0).tolist()
        if len(parents) > parent_cap:
            raise ValueError(f"parent count {len(parents)} exceeds cap {parent_cap}")
        for slot, parent_idx in enumerate(parents):
            block[dst, slot] = 1.0
            if mode == "lag1":
                block[dst, parent_cap + slot] = float(values[parent_idx])
            else:
                label = int(values[parent_idx])
                if not 0 <= label < num_classes:
                    raise ValueError(f"invalid label {label} for parent {parent_idx}")
                offset = parent_cap + slot * num_classes + label
                block[dst, offset] = 1.0
    return block


@dataclass
class GroupSample:
    group_id: str
    target_date: str
    x: np.ndarray
    y: np.ndarray
    parent_feat: np.ndarray
    dag_adj: np.ndarray
    city_names: List[str]


def load_group_samples(
    data_dir: str,
    split: str,
    num_classes: int,
    parent_cap: int,
    parent_mode: str,
    parent_predictions_dir: str = "",
    max_groups: int = 0,
) -> List[GroupSample]:
    rows = load_group_rows(data_dir, split)
    if max_groups > 0:
        rows = rows[:max_groups]
    samples: List[GroupSample] = []
    for row in rows:
        raw_ts = np.load(row["raw_ts_path"]).astype(np.float32)
        labels = np.load(row["labels_path"]).astype(np.int64)
        dag_adj = np.load(row["dag_adj_path"]).astype(np.float32)
        with open(row["city_names_path"], "r", encoding="utf-8") as stream:
            city_names = json.load(stream)

        x = np.transpose(raw_ts, (1, 0, 2)).astype(np.float32)
        if parent_mode == "lag1":
            parent_values = raw_ts[:, -1, 0].astype(np.float32)
        elif parent_mode == "predicted":
            prediction_path = os.path.join(parent_predictions_dir, split, f"{row['group_id']}.npy")
            parent_values = np.load(prediction_path).astype(np.int64)
            if parent_values.shape != labels.shape:
                raise ValueError(f"prediction shape mismatch for {row['group_id']}: {parent_values.shape} vs {labels.shape}")
        else:
            parent_values = labels
        parent_feat = build_parent_block(
            values=parent_values,
            dag_adj=dag_adj,
            num_classes=num_classes,
            parent_cap=parent_cap,
            mode=parent_mode,
        )
        samples.append(
            GroupSample(
                group_id=row["group_id"],
                target_date=row.get("target_date", row["group_id"]),
                x=x,
                y=labels,
                parent_feat=parent_feat,
                dag_adj=dag_adj,
                city_names=city_names,
            )
        )
    return samples


class StandardScaler4D:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=(0, 1, 2), keepdims=True)
        self.std = x.std(axis=(0, 1, 2), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class GroupBundleDataset(Dataset):
    def __init__(self, samples: List[GroupSample], scaler: StandardScaler4D):
        self.group_ids = [sample.group_id for sample in samples]
        self.target_dates = [sample.target_date for sample in samples]
        self.city_names = [sample.city_names for sample in samples]
        x = np.stack([sample.x for sample in samples], axis=0).astype(np.float32)
        parent_feat = np.stack([sample.parent_feat for sample in samples], axis=0).astype(np.float32)
        y = np.stack([sample.y for sample in samples], axis=0).astype(np.int64)
        self.x = torch.from_numpy(scaler.transform(x)).float()
        self.parent_feat = torch.from_numpy(parent_feat).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.parent_feat[idx], self.y[idx]


class MGSFormerOracleParentsClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int = 8,
        node_num: int = 13,
        feature_dim: int = 13,
        parent_dim: int = 21,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_class: int = 6,
        use_parent_features: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.node_num = node_num
        self.feature_dim = feature_dim
        self.parent_dim = parent_dim
        self.hidden_dim = hidden_dim
        self.use_parent_features = use_parent_features
        self.half_len = max(1, seq_len // 2)

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.temporal_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if use_parent_features:
            self.parent_proj = nn.Sequential(
                nn.LayerNorm(parent_dim),
                nn.Linear(parent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.parent_proj = None
            self.fusion = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class),
        )

    def forward(self, x: torch.Tensor, parent_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, node_num, feature_dim = x.shape
        if seq_len != self.seq_len or node_num != self.node_num or feature_dim != self.feature_dim:
            raise ValueError(
                f"expected input [{self.seq_len}, {self.node_num}, {self.feature_dim}], "
                f"got [{seq_len}, {node_num}, {feature_dim}]"
            )

        h = self.input_proj(x)
        h_t = h.permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, self.hidden_dim)
        attn_t, _ = self.temporal_attn(h_t, h_t, h_t)
        h_t = h_t + attn_t
        h_t = h_t + self.temporal_ffn(h_t)
        h_t = h_t.reshape(batch_size, node_num, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        s_full = h_t.mean(dim=1)
        s_half = h_t[:, -self.half_len :].mean(dim=1)
        s_last = h_t[:, -1]
        node_feat = self.scale_mix(torch.cat([s_full, s_half, s_last], dim=-1))

        attn_s, _ = self.spatial_attn(node_feat, node_feat, node_feat)
        node_feat = node_feat + attn_s
        node_feat = node_feat + self.spatial_ffn(node_feat)

        if self.use_parent_features:
            parent_emb = self.parent_proj(parent_feat)
            node_feat = self.fusion(torch.cat([node_feat, parent_emb], dim=-1))

        return self.classifier(node_feat)


def compute_class_weight(y_train: np.ndarray, num_classes: int = 6) -> torch.Tensor:
    flat = y_train.reshape(-1)
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    total_loss = 0.0
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    with torch.no_grad():
        for xb, parent_b, yb in loader:
            xb = xb.to(device)
            parent_b = parent_b.to(device)
            yb = yb.to(device)
            logits = model(xb, parent_b)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            y_true_all.append(yb.detach().cpu().numpy().reshape(-1))
            y_pred_all.append(pred.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    precision_macro, recall_macro, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    precision_micro, recall_micro, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="micro", zero_division=0
    )
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    per_class = {}
    for i in range(num_classes):
        per_class[str(i)] = {
            "label_name": LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i),
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "support": int(s_cls[i]),
        }
    return {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(macro_f1),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "micro_f1": float(micro_f1),
        "loss": float(total_loss / max(len(loader.dataset), 1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
        "per_class": per_class,
    }


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for xb, parent_b, yb in loader:
        xb = xb.to(device)
        parent_b = parent_b.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, parent_b)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return float(total_loss / max(len(loader.dataset), 1))


def parse_args():
    parser = argparse.ArgumentParser(description="MGSFormer parent-context controlled experiment")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--parent_cap", type=int, default=3)
    parser.add_argument("--max_train_groups", type=int, default=0)
    parser.add_argument("--max_eval_groups", type=int, default=0)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--disable_parent_features", action="store_true")
    parser.add_argument("--parent_mode", choices=("oracle", "none", "lag1", "predicted"), default="oracle")
    parser.add_argument("--parent_predictions_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    use_parent_features = not args.disable_parent_features and args.parent_mode != "none"
    if args.parent_mode == "predicted" and not args.parent_predictions_dir:
        raise ValueError("--parent_predictions_dir is required for --parent_mode predicted")

    train_samples = load_group_samples(
        data_dir=args.data_dir,
        split=args.train_split,
        num_classes=args.num_classes,
        parent_cap=args.parent_cap,
        parent_mode=args.parent_mode,
        parent_predictions_dir=args.parent_predictions_dir,
        max_groups=args.max_train_groups,
    )
    val_samples = load_group_samples(
        data_dir=args.data_dir,
        split=args.val_split,
        num_classes=args.num_classes,
        parent_cap=args.parent_cap,
        parent_mode=args.parent_mode,
        parent_predictions_dir=args.parent_predictions_dir,
        max_groups=args.max_eval_groups,
    )
    test_samples = load_group_samples(
        data_dir=args.data_dir,
        split=args.test_split,
        num_classes=args.num_classes,
        parent_cap=args.parent_cap,
        parent_mode=args.parent_mode,
        parent_predictions_dir=args.parent_predictions_dir,
        max_groups=args.max_eval_groups,
    )

    train_x = np.stack([sample.x for sample in train_samples], axis=0).astype(np.float32)
    scaler = StandardScaler4D().fit(train_x)
    train_ds = GroupBundleDataset(train_samples, scaler)
    val_ds = GroupBundleDataset(val_samples, scaler)
    test_ds = GroupBundleDataset(test_samples, scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    example_x, example_parent, example_y = train_ds[0]
    print(f"device={device}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"x shape={tuple(example_x.shape)} parent shape={tuple(example_parent.shape)} y shape={tuple(example_y.shape)}")
    print(f"use_parent_features={use_parent_features}")

    model = MGSFormerOracleParentsClassifier(
        seq_len=int(example_x.shape[0]),
        node_num=int(example_x.shape[1]),
        feature_dim=int(example_x.shape[2]),
        parent_dim=int(example_parent.shape[1]),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_class=args.num_classes,
        use_parent_features=use_parent_features,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.weighted_loss:
        class_weight = compute_class_weight(train_ds.y.numpy(), num_classes=args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        print(f"class_weight={class_weight.detach().cpu().numpy().tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["resolved_device"] = str(device)
    config["use_parent_features"] = use_parent_features
    config["train_groups"] = len(train_ds)
    config["val_groups"] = len(val_ds)
    config["test_groups"] = len(test_ds)
    save_json(os.path.join(args.out_dir, "run_config.json"), config)

    best_metric = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion, num_classes=args.num_classes)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_micro_f1={val_metrics['micro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))
            save_json(os.path.join(args.out_dir, "best_val_metrics.json"), val_metrics)

    save_json(os.path.join(args.out_dir, "train_history.json"), history)

    if best_state is None:
        raise RuntimeError("training finished without a best checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, criterion, num_classes=args.num_classes)
    save_json(os.path.join(args.out_dir, "test_metrics.json"), test_metrics)
    print(
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"test_micro_f1={test_metrics['micro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
