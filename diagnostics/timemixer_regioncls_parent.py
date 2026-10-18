#!/usr/bin/env python3
"""Formal TimeMixerStyleClassifier with controlled parent-context inputs."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timemixer_regioncls_formal import (  # noqa: E402
    TimeMixerStyleClassifier,
    compute_class_weights,
    metrics_from_arrays,
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parent_block(values, dag, mode, cap=12, classes=6):
    n = len(values)
    width = cap * 2 if mode == "lag1" else cap + cap * classes
    block = np.zeros((n, width), np.float32)
    for child in range(n):
        parents = np.flatnonzero(dag[child] > 0).tolist()
        if len(parents) > cap:
            raise ValueError(f"parent count {len(parents)} exceeds cap {cap}")
        for slot, parent in enumerate(parents):
            block[child, slot] = 1.0
            if mode == "lag1":
                block[child, cap + slot] = float(values[parent])
            else:
                block[child, cap + slot * classes + int(values[parent])] = 1.0
    return block


class ParentDataset(Dataset):
    def __init__(self, obj, mode, predictions=None, mean=None, std=None):
        self.x = obj["x"].astype(np.float32).copy()
        self.y = obj["y"].astype(np.int64).copy()
        self.dag = obj["dag_adj"].astype(np.float32).copy()
        if mean is not None:
            self.x = (self.x - mean) / std
        features = []
        for i in range(len(self.x)):
            if mode == "lag1":
                # Use the unnormalized final input-day parent PM2.5.
                values = obj["x"][i, -1, :, 0]
            elif mode == "oracle":
                values = self.y[i]
            elif mode == "predicted":
                values = predictions[i]
            else:
                raise ValueError(mode)
            features.append(parent_block(values, self.dag[i], mode))
        self.parent = np.stack(features).astype(np.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.parent[i]), torch.from_numpy(self.y[i])


class ParentTimeMixer(TimeMixerStyleClassifier):
    def __init__(self, parent_dim, **kwargs):
        super().__init__(**kwargs)
        self.parent_proj = nn.Sequential(nn.LayerNorm(parent_dim), nn.Linear(parent_dim, self.d_model), nn.GELU())
        self.parent_fusion = nn.Sequential(nn.LayerNorm(self.d_model * 2), nn.Linear(self.d_model * 2, self.d_model), nn.GELU(), nn.Dropout(0.1))

    def forward(self, history_data, parent_feat):
        pm25 = history_data[..., 0]
        time_context = self.time_embed(self._build_time_mark(history_data)).mean(dim=1)
        reps = []
        for scale_x, mlp in zip(self._multi_scale_pm25(pm25), self.scale_mlps):
            reps.append(mlp(scale_x.permute(0, 2, 1)))
        rep = torch.cat(reps, dim=-1)
        time_context = time_context.unsqueeze(1).expand(-1, self.num_nodes, -1)
        rep = self.node_mixer(torch.cat([rep, time_context], dim=-1))
        rep = self.parent_fusion(torch.cat([rep, self.parent_proj(parent_feat)], dim=-1))
        return self.classifier(rep)


@torch.no_grad()
def base_predictions(data_obj, checkpoint, device, mean, std):
    x = torch.from_numpy(((data_obj["x"].astype(np.float32) - mean) / std).astype(np.float32))
    model = TimeMixerStyleClassifier(seq_len=8, num_nodes=x.shape[2], num_classes=6, d_model=64, rep_len=8, moving_avg=3, dropout=0.1, down_sampling_window=2, down_sampling_layers=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)); model.eval()
    pred = []
    for start in range(0, len(x), 64):
        xb = x[start:start+64].to(device)
        pred.append(model(xb).argmax(-1).cpu().numpy())
    return np.concatenate(pred)


def evaluate(model, loader, criterion, device):
    model.eval(); total=0; count=0; ys=[]; ps=[]
    with torch.no_grad():
        for x,p,y in loader:
            x,p,y=x.to(device),p.to(device),y.to(device); logits=model(x,p)
            total += criterion(logits.reshape(-1,6),y.reshape(-1)).item()*y.numel(); count += y.numel()
            ys.append(y.cpu().numpy().ravel()); ps.append(logits.argmax(-1).cpu().numpy().ravel())
    m=metrics_from_arrays(np.concatenate(ys),np.concatenate(ps)); m["loss"]=total/count; return m


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data_dir",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--mode",choices=("oracle","lag1","predicted"),required=True); ap.add_argument("--base_checkpoint",default="")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--patience",type=int,default=8)
    ap.add_argument("--device",default="cuda"); args=ap.parse_args(); set_seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    objs={s:np.load(Path(args.data_dir)/f"{s}.npz",allow_pickle=True) for s in ("train","val","test")}
    mean=objs["train"]["x"].mean(axis=(0,1,2),keepdims=True).astype(np.float32); std=objs["train"]["x"].std(axis=(0,1,2),keepdims=True).astype(np.float32); std[std<1e-6]=1
    preds={}
    if args.mode=="predicted":
        if not args.base_checkpoint: raise ValueError("predicted mode requires --base_checkpoint")
        for s in objs: preds[s]=base_predictions(objs[s],args.base_checkpoint,device,mean,std)
    ds={s:ParentDataset(objs[s],args.mode,preds.get(s),mean,std) for s in objs}
    dl={"train":DataLoader(ds["train"],64,shuffle=True,num_workers=0),"val":DataLoader(ds["val"],64,num_workers=0),"test":DataLoader(ds["test"],64,num_workers=0)}
    parent_dim=ds["train"].parent.shape[-1]
    model=ParentTimeMixer(parent_dim=parent_dim,seq_len=8,num_nodes=13,num_classes=6,d_model=64,rep_len=8,moving_avg=3,dropout=0.1,down_sampling_window=2,down_sampling_layers=2).to(device)
    criterion=nn.CrossEntropyLoss(weight=compute_class_weights(ds["train"].y).to(device)); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1; bad=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; count=0
        for x,p,y in dl["train"]:
            x,p,y=x.to(device),p.to(device),y.to(device); opt.zero_grad(); logits=model(x,p); loss=criterion(logits.reshape(-1,6),y.reshape(-1)); loss.backward(); opt.step(); total+=loss.item()*y.numel(); count+=y.numel()
        val=evaluate(model,dl["val"],criterion,device); history.append({"epoch":epoch,"train_loss":total/count,"val_macro_f1":val["macro_f1"],"val_micro_f1":val["micro_f1"]})
        print(f"epoch={epoch} train_loss={total/count:.6f} val_macro_f1={val['macro_f1']:.6f} val_micro_f1={val['micro_f1']:.6f}",flush=True)
        if val["macro_f1"]>best: best=val["macro_f1"]; bad=0; torch.save(model.state_dict(),out/"best_model.pt")
        else: bad+=1
        if bad>=args.patience: break
    model.load_state_dict(torch.load(out/"best_model.pt",map_location=device)); test=evaluate(model,dl["test"],criterion,device)
    (out/"final_result.json").write_text(json.dumps({"mode":args.mode,"best_val_macro_f1":best,"test":test,"history":history},indent=2),encoding="utf-8")
    print(f"test_accuracy={test['accuracy']:.6f} test_macro_f1={test['macro_f1']:.6f} test_micro_f1={test['micro_f1']:.6f}",flush=True)


if __name__=="__main__": main()
