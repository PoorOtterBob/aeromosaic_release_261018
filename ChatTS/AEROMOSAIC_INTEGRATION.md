# ChatTS integration in AeroMosaic

This directory is a curated source subset from the official ChatTS repository at commit `a16ca1a7bd2d0cbe1dd40af37cb5658e7008357e`. Every file that is present in both this subset and that upstream commit is byte-identical. Directories unrelated to AeroMosaic packaging, including upstream demos, evaluation data, figures and tests, are not vendored.

AeroMosaic does not patch ChatTS data-generation or vLLM utility code. Its paper-aligned training and evaluation entrypoints use Hugging Face Transformers with the external checkpoint `bytedance-research/ChatTS-8B` at revision `7216a84034063c2d394901fe45d20a1e8d773a1f`, plus AeroMosaic's own prefix, LoRA, pooling, DAG residual, fusion and classification code under `pipeline/`.

The following files previously differed from upstream during package staging and were restored exactly after audit because none of the differences was required by AeroMosaic:

- `chatts/align/mts_local_llm_qa.py`
- `chatts/align/uts_template_qa.py`
- `chatts/ts_generator/generate.py`
- `chatts/ts_generator/local_changes.py`
- `chatts/utils/inference_tsmllm_vllm.py`
- `README.md`

The source commit and model revision identify different artifacts: the former pins the vendored utility source, while the latter pins the actual pretrained checkpoint used by the paper.
