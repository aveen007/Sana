# Text-Embedding Adaptation Dataset

## Split protocol

The repository already contains a leakage-free split suitable for fitting and
evaluating a text-embedding projection:

| Purpose | Source | Rows | Unique prompts |
| --- | --- | ---: | ---: |
| Projection fit/validation | `diffusion/post_training/dataset/geneval/train_metadata.jsonl` | 50,000 | 33,199 |
| Official final evaluation | `tools/metrics/geneval/prompts/evaluation_metadata.jsonl` | 553 | 553 |
| Batched copy of official evaluation | `diffusion/post_training/dataset/geneval/test_metadata.jsonl` | 2,212 | 553 |

There is no prompt overlap between the 33,199 unique training prompts and the
553 official GenEval prompts. Do not fit the projection on either official-eval
file. The 2,212-row test file repeats each official prompt four times and should
not be used as extra training data.

The extraction script deterministically selects 10,000 fit prompts, holds out
another 1,000 training-source prompts for projection validation, and keeps the
official 553 prompts as a final evaluation-only split.

## Server-side extraction

Preview the split without loading the model or writing files:

```bash
python tools/extract_sana_text_embeddings.py \
  --output-dir output/text_embeddings/sana_gemma2_geneval \
  --fit-prompts 10000 \
  --validation-prompts 1000 \
  --dry-run
```

Run the initial compact export:

```bash
python tools/extract_sana_text_embeddings.py \
  --output-dir output/text_embeddings/sana_gemma2_geneval_pooled \
  --fit-prompts 10000 \
  --validation-prompts 1000 \
  --representation pooled \
  --batch-size 4 \
  --shard-size 128 \
  --resume
```

This uses the existing environment and does not install, remove, or change
PyTorch. The encoder is loaded in bfloat16 exactly as in SANA inference, while
embeddings are stored in float16. Increase `--batch-size` only if GPU memory
allows it.

The pooled export should occupy roughly 120 MiB. It contains both a masked mean
and the last non-padding token vector. It is appropriate for developing a
vector-level projection, but pooled vectors cannot directly replace SANA's
token-level cross-attention conditioning.

If the final adapter will generate SANA's complete conditioning sequence, use
a separate directory and export the exact token-level tensors:

```bash
python tools/extract_sana_text_embeddings.py \
  --output-dir output/text_embeddings/sana_gemma2_geneval_tokens \
  --fit-prompts 10000 \
  --validation-prompts 1000 \
  --representation tokens \
  --batch-size 4 \
  --shard-size 128 \
  --resume
```

That token export is much larger: approximately 15 GiB for the configured
11,553 prompts. Use `--representation both` only when both forms are genuinely
needed. Use `--fit-prompts 0` to select all remaining unique training prompts.

## Output contract

The output directory contains:

- `run_config.json`: selection and encoder configuration, including the exact
  SANA CHI prompt prefix.
- `fit_prompts.jsonl`, `validation_prompts.jsonl`, and
  `official_eval_prompts.jsonl`: ordered prompts for encoding in the notebook.
- `null_conditioning.safetensors`: the exact unconditional 300-token embedding
  used for classifier-free guidance.
- Sharded Safetensors files named after their split and row interval.
- `manifest.json`: completed shard index and hidden size.

Every shard contains `row_indices`, `input_ids`, and `attention_mask`. Pooled
exports add `pooled_mean` and `pooled_last`; token exports add `hidden_states`
with shape `[batch, 300, 2304]` for the Sana-0.6B configuration.

In the notebook, encode the prompts from the three JSONL manifests in their
stored order. Fit only on `fit`, choose projection settings using `validation`,
and report GenEval only once on `official_eval`.
