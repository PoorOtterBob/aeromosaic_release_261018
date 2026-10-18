# Baseline Pipeline Detail: A+C+T

> **Historical record, not a current release recipe.** This note describes an April 2026 experiment tree whose scripts and data paths are not included here. For runnable current AeroMosaic procedures, see [`docs/reproduction.md`](reproduction.md); baseline implementation provenance and the remaining method-by-method gaps are in [`docs/baseline_reproduction.md`](baseline_reproduction.md).

## 1. Baseline 定义

本文 baseline 采用 A+C+T 配置：

- A: 历史 PM2.5（`raw_ts` 第 0 列）
- C: 降水文本 `precip_text`（保留在 prompt 中）
- T: 时间特征（`raw_ts` 第 11-12 列）
- 移除 B: 历史天气数值特征（`raw_ts` 第 1-10 列）

因此数值输入通过 `ablation_mask=0,11,12` 保留 PM2.5 与时间列，天气列置零；文本输入继续保留降水描述。

## 2. 总体运行链路

```text
precip bundle
  -> train_balanced_gc_ablation.py
  -> train_dagresidual_ablation.py
  -> final_adapter
  -> eval_dagresidual_ablation.py
  -> summary.json + predictions.csv + per_label_*.npy
  -> collect_seeds_results.py
  -> aggregate_results.json / csv / mean-std npy
```

统一调度入口：

```text
ChatTS-main/ChatTS-main/ablation/run_baseline_abl_ac.sh
```

## 3. 输入格式

每个 group 样本由 `group_index.jsonl` 索引到以下文件：

- `prompts_path`: prompt 列表或含 `prompt` 字段的字典列表
- `raw_ts_path`: `float32`, shape `(N, 8, 13)`
- `labels_path`: `int64`, shape `(N,)`
- `dag_adj_path`: `float32`, shape `(N, N)`
- `dag_level_path`: `int64`, shape `(N,)`
- `is_root_path`: `int64`, shape `(N,)`
- `parent_count_path`: `int64`, shape `(N,)`

其中：

- 第 0 列为 PM2.5 历史
- 第 1-10 列为天气数值特征
- 第 11-12 列为时间特征

## 4. 训练配置

baseline 固定参数：

- `--bf16`
- `--ablation_mask 0,11,12`
- `--dag_residual_enable`
- `--dag_residual_alpha_init -3.0`
- `--dag_residual_hidden_dim 256`
- `--root_aware_loss`
- `--root_weight 1.5`
- `--root_label_extra 3:2.0,4:3.0`
- `--max_root_sample_weight 4.0`
- `--root_region_env_fusion`
- `--root_region_env_weight 0.8`
- `--root_region_exclude_self`
- `--num_train_epochs 3`
- `--learning_rate 1e-4`
- `--gradient_accumulation_steps 2`
- `--logging_steps 10`
- LoRA 开启

## 5. 输出格式

### 单 seed 训练输出

- `final_adapter/`
- `train_metrics.json`
- `eval_metrics.json`
- `trainer_state_summary.json`
- `run_config.json`

### 单 seed 测试输出

- `test/predictions.csv`
- `test/summary.json`
- `test/per_label_precision.npy`
- `test/per_label_recall.npy`
- `test/per_label_f1.npy`
- `summary_all.json`

### 区域级 5-seed 汇总输出

- `aggregate_results.json`
- `aggregate_results.csv`
- `per_label_precision_mean.npy`
- `per_label_precision_std.npy`
- `per_label_recall_mean.npy`
- `per_label_recall_std.npy`
- `per_label_f1_mean.npy`
- `per_label_f1_std.npy`

## 6. 论文报告建议

最终论文按区域报告：

- Macro-F1 mean ± std
- Micro-F1 mean ± std
- Accuracy mean ± std
- Per-label precision / recall / f1 的 mean ± std

这样可直接覆盖 6 区域 × 5 seeds 的 baseline 统计要求。
