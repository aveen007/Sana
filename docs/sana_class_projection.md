# SANA class-prototype projection

This pipeline follows the OV-DQUO calculation-set design without placing any
official GenEval prompt or object label in the projector calculation set.

## Data layout

- Calculation dictionary: 3,000 WordNet noun classes by default.
- Prompt augmentation: the first 5 OV-DQUO/ImageNet templates by default. More
  templates are supported, but increase encoder work linearly.
- Calculation row: one class prototype, obtained by averaging normalized
  class-token hidden states across its templates.
- Benchmark: the original 553 GenEval prompts, exported separately and in
  their original order.

The 15,000 templated strings are intermediate encoder inputs. They produce
3,000 calculation rows; they are not treated as independent classes. Class
prototype extraction dynamically pads each batch rather than evaluating all
300 output positions. The official benchmark export still uses the exact
fixed 300-position SANA representation.

## Server commands

The class-list builder uses NLTK WordNet but does not require changing the
PyTorch installation:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate samaudio
cd /home/tnn/shared/sana

python -m pip install --no-deps nltk
python -m nltk.downloader wordnet omw-1.4

python tools/build_sana_projection_class_set.py \
  --num-classes 3000 \
  --num-templates 5 \
  --output-dir output/text_embeddings/sana_class_projection/spec_5templates

python tools/extract_sana_class_prototypes.py \
  --class-spec output/text_embeddings/sana_class_projection/spec_5templates/class_set.json \
  --output-dir output/text_embeddings/sana_class_projection/sana_5templates \
  --batch-size 8 \
  --class-shard-size 16 \
  --eval-shard-size 32 \
  --resume
```

Increase `--batch-size` only if GPU memory allows it. `--resume` validates and
keeps completed shards after an interrupted run. When extraction succeeds,
the shards are packed, reopened for verification, and removed automatically.
Use `--keep-shards` only when the intermediate files are needed for debugging.

If an older run already completed all shards, consolidate it without loading
Gemma or recomputing embeddings:

```bash
python tools/extract_sana_class_prototypes.py \
  --class-spec output/text_embeddings/sana_class_projection/spec_5templates/class_set.json \
  --output-dir output/text_embeddings/sana_class_projection/sana_5templates \
  --class-shard-size 16 \
  --eval-shard-size 32 \
  --pack-only
```

## Outputs

`sana_class_embeddings.pt` is the calculation file to download and contains:

- `class_names`: exactly 3,000 calculation classes;
- `templates`: the exact template strings and ordering;
- `embeddings`: `[3000, 2304]` natural-scale class prototypes;
- `class_spec`: the complete shared class/template/exclusion specification.

`projection_class_set.json` is a standalone copy of that same specification.
Upload either it or `sana_class_embeddings.pt` to Kaggle; the VLM extractor
must read `class_names` and `templates` from it instead of rebuilding them.

`sana_official_geneval_embeddings.pt` contains untouched native SANA data:

- `embeddings`: `[553, 300, 2304]`;
- `attention_mask`: `[553, 300]`;
- `prompt_texts` and `ids`: the exact official prompt order and hashes;
- `metadata`: the complete official GenEval rows.

The packed files and their SHA-256 hashes are recorded in `manifest.json`.

## Projected GenEval bundle

The final notebook output consumed by `scripts/inference_geneval.py` must use:

```text
Y_pred:        [553, 300, 2304] or [553, 1, 300, 2304]
attention_mask:[553, 300]
prompt_texts:  list of 553 official prompts
ids:           SHA-256 identifiers for those prompts
```

This native-sequence form is passed directly to SANA. It is not pooled and no
embedding vector is repeated across the 300 positions.
