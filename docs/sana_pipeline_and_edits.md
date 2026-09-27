# SANA image pipeline and projection edit map

This note describes the exact path used by the 600M/1024px GenEval run. The
core SANA transformer was not modified for the projection experiment. The
mapped tensor is substituted for Gemma's output in the GenEval inference
script, before it enters the original frozen SANA model.

## End-to-end path

```text
GenEval prompt
  -> CHI prefix + prompt
  -> Gemma-2-2B-IT tokenizer and decoder
  -> native text tensor [B, 1, 300, 2304]
       OR
     mapped tensor loaded from Y_pred [B, 1, 300, 2304]  <-- our injection
  -> CaptionEmbedder MLP: 2304 -> 1152 -> GELU -> 1152
  -> attention_y_norm (RMSNorm; configured scale factor 0.01)
  -> keep active positions using the 300-position attention mask
  -> 28 x SanaMSBlock
       1. image-token linear self-attention
       2. text cross-attention: Q from image tokens, K/V from text tokens
       3. GLUMBConv feed-forward network
  -> final layer + unpatchify
  -> repeat denoising for 20 Flow-DPM-Solver steps
  -> DC-AE decoder
  -> 1024px image
```

The mapped and native branches merge at `caption_embs`. Everything below that
point uses the same checkpoint weights and the same model code.

## Base SANA code

| Purpose | File and location | What it does |
|---|---|---|
| Experiment configuration | `configs/sana_config/1024ms/Sana_600M_img1024.yaml:18` | Selects `SanaMS_600M_P1_D28`, Gemma-2-2B-IT, 300 text positions, 1024px images, linear self-attention and GLUMBConv. |
| Text encoder construction | `diffusion/model/builder.py:120` | Loads the Gemma tokenizer with right padding and obtains the Gemma decoder. |
| Model argument construction | `diffusion/utils/config.py:448` | Passes 300 positions, 2304 caption channels, normalization settings and attention settings into SANA. |
| 600M model factory | `diffusion/model/nets/sana_multi_scale.py:502` | Creates the 28-block, width-1152, patch-size-1 model. |
| Model components | `diffusion/model/nets/sana_multi_scale.py:244` | Builds the latent patch embedder, caption embedder, 28 blocks and final layer. |
| Main transformer forward | `diffusion/model/nets/sana_multi_scale.py:323` | Embeds image latents and text, applies RMSNorm/masking, executes all blocks, then unpatchifies. |
| One transformer block | `diffusion/model/nets/sana_multi_scale.py:54` | Defines image self-attention, text cross-attention and the image FFN. Its forward order is at line 145. |
| Text projection | `diffusion/model/nets/sana_blocks.py:1097` | `CaptionEmbedder`; its MLP call is at line 1150. The MLP is `fc1 -> GELU -> fc2`. |
| Cross-attention | `diffusion/model/nets/sana_blocks.py:48` | Produces Q from image tokens and K/V from text conditioning at lines 83-110. |
| Latent decoding | `diffusion/model/builder.py:415` | Converts the final denoised latent into an image with DC-AE. |

### What one `SanaMSBlock` does

For block input image tokens `x`, text conditioning `y`, and timestep embedding
`t`, the original block is:

```text
x = x + gated_linear_self_attention(norm(x), t)
x = x + cross_attention(query=x, key=y, value=y)
x = x + gated_GLUMBConv_FFN(norm(x), t)
```

Cross-attention uses learned projections:

```text
Q = q_linear(image tokens)
[K, V] = kv_linear(text tokens)
attention = softmax(Q K^T / sqrt(head_dim)) V
```

For this checkpoint, SANA hidden width is 1152 with 16 cross-attention heads,
so each cross-attention head has width 72. The same processed text tensor is
sent to every block, but every block has its own learned `kv_linear` weights.

## Where our mapped tensor is injected

The only behavior-changing injection is in `scripts/inference_geneval.py`:

1. `load_projected_text_embeddings()` at line 82 loads and validates `Y_pred`,
   prompt order, IDs, tensor shape and attention mask.
2. Lines 291-313 construct the normal SANA prompt and native Gemma tensor.
3. Lines 314-337 select either the native tensor or our mapped tensor and mask.
4. Lines 394-454 pass the selected `caption_embs` to the unmodified sampler and
   `model.forward_with_dpmsolver`.

The effective branch is:

```python
if projected_text_embeddings is None:
    caption_embs = native_caption_embs
    emb_masks = native_emb_masks
else:
    caption_embs = projected_text_embeddings[index]  # [300, 2304]
    emb_masks = projected_text_attention_masks[index]
```

Thus, the projector replaces the output of Gemma. It does **not** replace
`CaptionEmbedder`, RMSNorm, K/V projections, attention, diffusion blocks, the
sampler, or the decoder.

The unconditional classifier-free-guidance tensor (`null_y`) remains the
normal SANA/Gemma empty-prompt embedding.

## Our added files and edits

| File | Role | Changes model behavior? |
|---|---|---|
| `scripts/inference_geneval.py` | Loads and injects projected conditioning; optionally invokes the debugger. | Yes, only when `--projected_text_embeddings` is supplied. |
| `tools/build_sana_llava_geneval_projection.py` | Offline LLaVA token extraction, alignment, projection and packing into native `[553,300,2304]` layout. | No; preprocessing only. |
| `diffusion/utils/conditioning_debugger.py` | Compares native/mapped tensors after caption MLP, RMSNorm, per-block K/V, fixed-Q attention and denoiser output. | No; instrumentation only. |
| `scripts/debug_conditioning_bos_rightpad.sh` | Convenience launcher for one-prompt debug. | No. |
| `scripts/run_geneval_bos_rightpad.sh` | Convenience launcher for projected GenEval. | Only by enabling the injection flag. |
| `scripts/*geneval.sh` wrappers | Forward the projected-file argument and preserve GPU selection. | No model changes. |
| `tools/extract_sana_text_embeddings.py` and `tools/extract_sana_class_prototypes.py` | Export native SANA targets used to fit/check the projector. | No; data export only. |
| `tests/test_conditioning_debugger.py` | Tests debugger calculations and safety. | No. |

No file under `diffusion/model/` differs from the repository state immediately
before our work (`38d8f6f`). The model block definitions and checkpoint path
therefore remain the base SANA implementation.

## Debugger insertion

The debugger call is at `scripts/inference_geneval.py:354`. It receives both:

```text
native_caption_embs  = actual Gemma output
caption_embs         = mapped tensor selected for generation
```

It then applies the loaded model's real modules to both tensors. The reusable
entry point is `debug_conditioning()` in
`diffusion/utils/conditioning_debugger.py:2291`; the main implementation is
`SanaConditioningDebugger` at line 1531.

It is bypassed unless `--conditioning_debug=true` is set. With
`--conditioning_debug_only=true`, it writes the report and stops before image
sampling.

## Why there are 300 positions

`300` is SANA's configured maximum text sequence length, not the number of
classes and not an embedding dimension. Each position has 2304 Gemma features,
so the model-facing text tensor is `[batch, 1, 300, 2304]`. The attention mask
marks only the real positions as active; padding positions are discarded before
cross-attention.

The inference script prepends SANA's CHI instruction, tokenizes with right
padding, then selects the BOS position plus the last 299 positions. The
corrected projector bundle reproduces that native position layout and mask.

## Useful commands for locating changes

```bash
# Everything changed since the repository state before our first edit
git diff --name-status 38d8f6f..HEAD

# See the exact injection patch
git diff 38d8f6f..HEAD -- scripts/inference_geneval.py

# Confirm that the original model implementation was not edited
git diff 38d8f6f..HEAD -- diffusion/model

# Find projection/debug integration points
rg -n "projected_text_embeddings|debug_conditioning" scripts diffusion tools
```
