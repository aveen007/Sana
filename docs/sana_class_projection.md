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
keeps completed shards after an interrupted run.

## Outputs

`sana_class_prototypes.pt` contains:

- `class_names`: exactly 3,000 calculation classes;
- `templates`: the exact template strings and ordering;
- `embeddings`: `[3000, 2304]` natural-scale class prototypes;
- `unit_embeddings`: `[3000, 2304]` normalized prototypes;
- `mean_norms`: the average native SANA class-token norm per class.

The `official-eval-*.safetensors` shards contain untouched native SANA data:

- `hidden_states`: `[shard_rows, 300, 2304]`;
- `attention_mask`: `[shard_rows, 300]`;
- `input_ids`: `[shard_rows, 300]`;
- `offset_mapping`: `[shard_rows, 300, 2]`;
- `row_indices`: positions in the official 553-prompt benchmark.

The shared `class_set.json` must also be used when extracting LLaVA class
prototypes so both encoders see the same classes, templates, and ordering.

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
