#!/usr/bin/env python3
"""Measure AeroMosaic forward-pass parameters, profiler FLOPs/MACs, and batch/s."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from train_yrd_graph_cls_v2_rootregion_best_dagresidual import DagGroupDataset, DagGroupCollator, RawTsPrefixSequenceClassifier

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--oracle-data", required=True)
p.add_argument("--noparent-data", required=True)
p.add_argument("--output", required=True)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--repeats", type=int, default=10)
a = p.parse_args()

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, use_fast=False)
tokenizer.truncation_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = RawTsPrefixSequenceClassifier(
    lm=get_peft_model(base, lora), raw_ts_shape=(8, 13), prefix_tokens=4,
    raw_ts_hidden_dim=512, num_labels=6, classifier_dropout=0.1,
    pooling_mode="mean", dag_residual_enable=True, dag_residual_hidden_dim=256,
    dag_residual_alpha_init=-3.0, dag_residual_dropout=0.0, dag_residual_norm="indegree",
).to(device).eval()
params_total = sum(x.numel() for x in model.parameters())
params_trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)

def run_model(batch: dict):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(**batch)

def inputs(data_dir: str):
    ds = DagGroupDataset(data_dir=data_dir, split_name="test", tokenizer=tokenizer, raw_ts_shape=(8,13), max_input_length=512, use_chat_template=False, root_region_env_fusion=True, root_region_env_weight=0.8, root_region_exclude_self=True)
    batch = DagGroupCollator(tokenizer)([ds[0]])
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

@torch.inference_mode()
def measure(name: str, batch: dict):
    for _ in range(a.warmup):
        run_model(batch)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_flops=True, record_shapes=False) as prof:
        run_model(batch)
        torch.cuda.synchronize()
    flops = sum(event.flops for event in prof.key_averages() if event.flops)
    torch.cuda.synchronize(); start=time.perf_counter()
    for _ in range(a.repeats):
        run_model(batch)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    return {
        "protocol": name,
        "batch_definition": "one BTH region-date group",
        "cities_per_batch": int(batch["input_ids"].shape[0]),
        "prompt_tokens_max": int(batch["input_ids"].shape[1]),
        "profiler_flops_per_batch": int(flops),
        "macs_per_batch_assuming_2_flops_per_mac": int(flops // 2),
        "repeats": a.repeats,
        "elapsed_seconds": elapsed,
        "batches_per_second": a.repeats / elapsed,
    }

result = {
    "device": torch.cuda.get_device_name(device),
    "torch": torch.__version__,
    "params_total": params_total,
    "params_trainable": params_trainable,
    "flops_method": "torch.profiler(with_flops=True); MACs reported as FLOPs / 2",
    "results": [measure("Oracle", inputs(a.oracle_data)), measure("NoParent", inputs(a.noparent_data))],
}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
