
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'pipeline'))
import eval_dagresidual_ablation as base

CONTEXT_RE = re.compile(r"Upstream parent-label context: .*? Predict the PM2\.5 risk class\.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path', required=True)
    p.add_argument('--adapter_dir', required=True)
    p.add_argument('--oracle_data_dir', required=True)
    p.add_argument('--noparent_data_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--eval_splits', default='test')
    p.add_argument('--raw_ts_shape', default='8,13')
    p.add_argument('--max_input_length', type=int, default=512)
    p.add_argument('--use_chat_template', action='store_true', default=False)
    p.add_argument('--trust_remote_code', action='store_true', default=False)
    p.add_argument('--bf16', action='store_true', default=False)
    p.add_argument('--load_in_4bit', action='store_true', default=False)
    # Keep this wrapper's namespace compatible with the current ablation evaluator.
    p.add_argument('--ablation_mask', type=str, default=None)
    p.add_argument('--no_lora', action='store_true', default=False)
    return p.parse_args()


def replace_parent_context(prompt: str, context: str) -> str:
    """Replace, never append to, the source bundle's upstream-label block."""
    start = prompt.find('Upstream graph context:')
    if start >= 0:
        boundaries = [
            prompt.find('Recent precipitation context:', start),
            prompt.find('Predict the final air-quality label', start),
        ]
        end = min((x for x in boundaries if x >= 0), default=len(prompt))
        return prompt[:start] + f'Upstream graph context:\n{context}\n' + prompt[end:]
    repl = f'Upstream parent-label context: {context} Predict the PM2.5 risk class.'
    if 'Upstream parent-label context:' in prompt:
        return CONTEXT_RE.sub(repl, prompt)
    return prompt + f' {repl}'


@torch.no_grad()
def predict_group_rollout(
    model,
    tokenizer,
    nop_row: Dict[str, Any],
    oracle_row: Dict[str, Any],
    raw_ts_shape,
    max_input_length: int,
    use_chat_template: bool,
    root_region_cfg: Dict[str, Any],
):
    city_names = base._load_json(Path(nop_row['city_names_path']), [])
    oracle_prompt_entries = base._load_json(Path(oracle_row['prompts_path']), [])
    oracle_prompts = [str(x.get('prompt', '')) if isinstance(x, dict) else str(x) for x in oracle_prompt_entries]

    raw_ts = np.load(nop_row['raw_ts_path']).astype(np.float32)
    labels = np.load(nop_row['labels_path']).astype(np.int64)
    dag_adj = np.load(nop_row['dag_adj_path']).astype(np.float32)
    dag_level = np.load(nop_row['dag_level_path']).astype(np.int64)
    is_root = np.load(nop_row['is_root_path']).astype(np.int64)
    parent_count_path = nop_row.get('parent_count_path')
    if parent_count_path:
        parent_count = np.load(parent_count_path).astype(np.int64)
    else:
        # Fallback for bundles that omit parent_count_path: derive indegree from DAG.
        parent_count = dag_adj.sum(axis=1).astype(np.int64)

    raw_ts = base.maybe_fuse_root_raw_ts(
        raw_ts=raw_ts,
        is_root=is_root,
        enable=bool(root_region_cfg.get('root_region_env_fusion', False)),
        weight=float(root_region_cfg.get('root_region_env_weight', 0.5)),
        exclude_self=bool(root_region_cfg.get('root_region_exclude_self', False)),
    )

    n = raw_ts.shape[0]
    current_contexts = ['unknown/no-parent'] * n
    fixed_preds: Dict[int, int] = {}
    fixed_logits: Dict[int, List[float]] = {}
    fixed_prompt_lens: Dict[int, int] = {}
    fixed_prompt_texts: Dict[int, str] = {}

    device = next(model.parameters()).device
    levels = sorted(int(x) for x in np.unique(dag_level).tolist())

    for level in levels:
        prompts = [replace_parent_context(oracle_prompts[i], current_contexts[i]) for i in range(n)]
        input_ids, attention_mask, prompt_lens = base.build_group_inputs(
            tokenizer=tokenizer,
            prompts=prompts,
            max_input_length=max_input_length,
            use_chat_template=use_chat_template,
        )
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            raw_ts=torch.tensor(raw_ts, dtype=torch.float32, device=device),
            dag_adj=torch.tensor(dag_adj, dtype=torch.float32, device=device),
            is_root=torch.tensor(is_root.astype(np.float32), dtype=torch.float32, device=device),
        )
        logits = outputs.logits.detach().float().cpu().numpy()
        preds = np.argmax(logits, axis=-1).astype(np.int64)

        for i in range(n):
            if int(dag_level[i]) == level and i not in fixed_preds:
                fixed_preds[i] = int(preds[i])
                fixed_logits[i] = [float(x) for x in logits[i].tolist()]
                fixed_prompt_lens[i] = int(prompt_lens[i])
                fixed_prompt_texts[i] = prompts[i]

        for child in range(n):
            if int(dag_level[child]) <= level:
                continue
            parents = np.flatnonzero(dag_adj[child]).tolist()
            if parents and all(int(p) in fixed_preds for p in parents):
                current_contexts[child] = ', '.join(f'{city_names[p]}: class {fixed_preds[int(p)]}' for p in parents)
            else:
                current_contexts[child] = 'unknown/no-parent'

    rows: List[Dict[str, Any]] = []
    parent_rows: List[Dict[str, Any]] = []
    for i in range(n):
        row = {
            'group_id': str(nop_row['group_id']),
            'target_date': str(nop_row.get('target_date', nop_row['group_id'])),
            'city_name': str(city_names[i]),
            'dag_level': int(dag_level[i]),
            'parent_count': int(parent_count[i]),
            'label': int(labels[i]),
            'pred': int(fixed_preds[i]),
            'correct': int(int(fixed_preds[i]) == int(labels[i])),
            'prompt_token_len': int(fixed_prompt_lens[i]),
            'is_root': int(is_root[i]),
            'prompt_text': fixed_prompt_texts[i],
        }
        for k in range(base.NUM_LABELS):
            row[f'logit_{k}'] = float(fixed_logits[i][k])
        rows.append(row)

        parents = np.flatnonzero(dag_adj[i]).tolist()
        for p in parents:
            parent_rows.append({
                'group_id': str(nop_row['group_id']),
                'target_date': str(nop_row.get('target_date', nop_row['group_id'])),
                'child_city': str(city_names[i]),
                'parent_city': str(city_names[p]),
                'child_dag_level': int(dag_level[i]),
                'true_parent_label': int(labels[p]),
                'pred_parent_label': int(fixed_preds[int(p)]),
                'correct': int(int(labels[p]) == int(fixed_preds[int(p)])),
            })
    return rows, parent_rows


def summarize_parent_rows(parent_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not parent_rows:
        return {'num_edges': 0, 'accuracy': 0.0, 'micro_f1': 0.0, 'macro_f1': 0.0, 'per_label': []}
    y_true = [int(x['true_parent_label']) for x in parent_rows]
    y_pred = [int(x['pred_parent_label']) for x in parent_rows]
    labels = sorted(set(y_true) | set(y_pred))
    summary = base.compute_metrics(y_true, y_pred, labels)
    summary['num_edges'] = len(parent_rows)
    summary['true_hist'] = {str(k): int(v) for k, v in sorted(Counter(y_true).items())}
    summary['pred_hist'] = {str(k): int(v) for k, v in sorted(Counter(y_pred).items())}
    return summary


def evaluate_split_rollout(split_name: str, noparent_data_dir: Path, oracle_data_dir: Path, model, tokenizer, raw_ts_shape, max_input_length: int, use_chat_template: bool, root_region_cfg: Dict[str, Any], out_dir: Path):
    base.ensure_dir(str(out_dir))
    nop_rows = [json.loads(line) for line in (noparent_data_dir / 'assets' / split_name / 'group_index.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    oracle_rows = [json.loads(line) for line in (oracle_data_dir / 'assets' / split_name / 'group_index.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    oracle_by_gid = {str(r['group_id']): r for r in oracle_rows}

    pred_rows: List[Dict[str, Any]] = []
    parent_rows: List[Dict[str, Any]] = []
    for nop_row in nop_rows:
        group_rows, group_parent_rows = predict_group_rollout(
            model=model,
            tokenizer=tokenizer,
            nop_row=nop_row,
            oracle_row=oracle_by_gid[str(nop_row['group_id'])],
            raw_ts_shape=raw_ts_shape,
            max_input_length=max_input_length,
            use_chat_template=use_chat_template,
            root_region_cfg=root_region_cfg,
        )
        pred_rows.extend(group_rows)
        parent_rows.extend(group_parent_rows)

    summary = base.summarize_rows(pred_rows)
    root_rows = [x for x in pred_rows if int(x['is_root']) == 1]
    nonroot_rows = [x for x in pred_rows if int(x['is_root']) == 0]
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_parent_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        by_level[str(int(row['dag_level']))].append(row)
        p = int(row['parent_count'])
        key = 'p0' if p == 0 else ('p1_2' if p <= 2 else ('p3_4' if p <= 4 else 'p5_plus'))
        by_parent_bin[key].append(row)
    slices = {}
    if root_rows:
        slices['root'] = base.summarize_rows(root_rows)
    if nonroot_rows:
        slices['nonroot'] = base.summarize_rows(nonroot_rows)
    slices['by_dag_level'] = {k: base.summarize_rows(v) for k, v in sorted(by_level.items()) if v}
    slices['by_parent_bin'] = {k: base.summarize_rows(v) for k, v in sorted(by_parent_bin.items()) if v}

    parent_summary = summarize_parent_rows(parent_rows)

    with (out_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['group_id','target_date','city_name','dag_level','parent_count','label','pred','correct','prompt_token_len','is_root','prompt_text'] + [f'logit_{i}' for i in range(base.NUM_LABELS)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pred_rows:
            writer.writerow(row)

    with (out_dir / 'parent_label_predictions.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['group_id','target_date','child_city','parent_city','child_dag_level','true_parent_label','pred_parent_label','correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in parent_rows:
            writer.writerow(row)

    payload = {'split': split_name, **summary, 'slices': slices, 'parent_label_summary': parent_summary}
    base.save_json(str(out_dir / 'summary.json'), payload)
    return payload


def main() -> None:
    args = parse_args()
    base.ensure_dir(args.out_dir)
    tokenizer, model, raw_ts_shape, root_region_cfg, dag_residual_cfg = base.load_tokenizer_and_model(args)
    split_names = [x.strip() for x in args.eval_splits.split(',') if x.strip()]
    all_summaries = []
    for split_name in split_names:
        out_dir = Path(args.out_dir) / split_name
        summary = evaluate_split_rollout(
            split_name=split_name,
            noparent_data_dir=Path(args.noparent_data_dir),
            oracle_data_dir=Path(args.oracle_data_dir),
            model=model,
            tokenizer=tokenizer,
            raw_ts_shape=raw_ts_shape,
            max_input_length=args.max_input_length,
            use_chat_template=args.use_chat_template,
            root_region_cfg=root_region_cfg,
            out_dir=out_dir,
        )
        all_summaries.append(summary)
        print('=' * 80)
        print(f"[{split_name}] num_samples = {summary['num_samples']}")
        print(f"[{split_name}] accuracy   = {summary['accuracy']:.6f}")
        print(f"[{split_name}] micro_f1   = {summary['micro_f1']:.6f}")
        print(f"[{split_name}] macro_f1   = {summary['macro_f1']:.6f}")
        print(f"[{split_name}] parent_acc = {summary['parent_label_summary']['accuracy']:.6f}")
        print(f"[{split_name}] parent_macro_f1 = {summary['parent_label_summary']['macro_f1']:.6f}")
    base.save_json(str(Path(args.out_dir) / 'summary_all.json'), {'summaries': all_summaries, 'dag_residual_config': dag_residual_cfg, 'root_region_config': root_region_cfg})


if __name__ == '__main__':
    main()
