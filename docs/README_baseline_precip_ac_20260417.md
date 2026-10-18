# Baseline A+C+T (原 abl_AC)

> **Historical record, not a current release recipe.** This note was copied from the April 2026 experiment workspace. Its commands and `proposal/` / `ChatTS-main/` paths are not included in this camera-ready package and must not be used to reproduce the current paper. Use [`docs/reproduction.md`](reproduction.md) for the current AeroMosaic protocol and [`docs/baseline_reproduction.md`](baseline_reproduction.md) for the baseline provenance and reproduction boundary.

本目录封装当前论文 baseline：保留历史 PM2.5 (A)、降水文本 (C) 和时间特征 (T)，移除历史天气数值特征 (B)。

## 配置定义

- 数值保留列：`ablation_mask=0,11,12`
- 数据：precip bundle
- 结构：DAG residual + LoRA + root-aware loss + root-region env fusion
- 精度：统一 `bf16`
- seeds：`42, 123, 456, 789, 2024`

## 区域数据映射

- YRD: `proposal/YRD_baseline/yrd_dag_group_bundle_precip`
- BTH: `proposal/BTH_baseline/bth_dag_group_bundle_precip`
- CYB: `proposal/CYB_baseline/cyb_dag_group_bundle_precip_short`
- ZHD: `proposal/ZHD_baseline/zhd_dag_group_bundle_precip`
- SDP: `proposal/SDP_baseline/sdp_dag_group_bundle_precip_short`
- CJD: `proposal/CJD_baseline/cjd_dag_group_bundle_precip`

## 运行方式

训练 + 测试评估入口：

```bash
bash ChatTS-main/ChatTS-main/ablation/run_baseline_abl_ac.sh
```

汇总 6 区域 5-seed 结果：

```bash
python proposal/baseline_precip_ac_20260417/collect_seeds_results.py
```

## 输出目录

默认输出目录：

```text
proposal/baseline_precip_ac_20260417/
```

每个 seed 的训练结果：

```text
<REGION>_seed<SEED>/
  final_adapter/
  train.log
  train_metrics.json
  eval_metrics.json
  trainer_state_summary.json
  run_config.json
```

每个 seed 的测试结果：

```text
<REGION>_seed<SEED>_eval_test/
  test/
    predictions.csv
    summary.json
    per_label_precision.npy
    per_label_recall.npy
    per_label_f1.npy
  summary_all.json
```

每个区域的聚合结果：

```text
<REGION>_aggregate/
  aggregate_results.json
  per_label_precision_mean.npy
  per_label_precision_std.npy
  per_label_recall_mean.npy
  per_label_recall_std.npy
  per_label_f1_mean.npy
  per_label_f1_std.npy
```

## 关键文件说明

- `summary.json`: 单个 seed 在 test split 上的 accuracy / micro_f1 / macro_f1 / per_label
- `predictions.csv`: 逐节点预测输出，含 `label/pred/logit_*`
- `per_label_*.npy`: 单个 seed 的 per-label precision / recall / f1
- `aggregate_results.json`: 区域级 5-seed mean ± std 汇总
