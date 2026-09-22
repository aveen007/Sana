# SANA Text-Embedding Projection: Experiments and Results

## What we did

1. Reproduced the native SANA-0.6B GenEval baseline: 553 prompts, 4 images per prompt, 1024 px, Flow-DPM-Solver, 20 steps, CFG 4.5.
2. Built LLaVA-to-SANA projectors and injected their `[300, 2304]` outputs at the same point as SANA's native Gemma text embeddings.
3. Tested a class/kernel projector and a token-aligned Nyström projector.
4. Instrumented SANA's real conditioning path and compared native versus mapped tensors at every meaningful transformation.
5. Found and fixed a tokenizer-padding mismatch in the token-aligned projection files.

## GenEval results

| Experiment | Overall | Correct images | Correct prompts | Status |
|---|---:|---:|---:|---|
| Native SANA baseline | **0.66732** | 65.19% | 78.30% | Valid |
| Initial projected embeddings | 0.01546 | 1.45% | 3.98% | Valid, very poor |
| Class/kernel projected native sequence | 0.02738 | 2.53% | 4.70% | Valid, very poor |
| Token-aligned projection, left-padded file | 0.00920 | 0.81% | 2.35% | Padding affected debugging, not packed conditioning |
| Token-aligned projection, right-padded file | **0.00972** | 0.86% | 2.53% | Valid; effectively unchanged |

### Task breakdown

| Task | Native baseline | Class/kernel projection | Token-aligned right-padded |
|---|---:|---:|---:|
| Counting | 70.00% | 0.00% | 0.00% |
| Color attribute | 40.00% | 0.00% | 0.00% |
| Colors | 87.23% | 4.26% | 0.27% |
| Position | 25.50% | 0.00% | 0.00% |
| Single object | 99.38% | 10.66% | 5.31% |
| Two objects | 78.28% | 1.52% | 0.25% |

## Projection-fit metrics

| Projector | Correct cosine | Wrong cosine | Separation gap | Top-1 | GenEval |
|---|---:|---:|---:|---:|---:|
| Class/kernel | 0.999998 | 0.608208 | 0.391790 | 1.000 | 0.02738 |
| Token-aligned Nyström | 0.978977 | 0.574295 | 0.404682 | 0.2528 | 0.00920 left-padded / 0.00972 right-padded |

High fit cosine therefore did **not** mean the mapped representation remained equivalent inside SANA.

## Actual SANA conditioning path

```text
Gemma sequence [300, 2304]
  -> CaptionEmbedder fc1 [2304 -> 1152]
  -> GELU
  -> fc2 [1152 -> 1152]
  -> RMSNorm
  -> attention-mask packing
  -> per-block cross-attention K/V projections
  -> image-latent conditioning
```

SANA does not average the 300 token positions into one class vector. They are a padded token sequence; only positions selected by the attention mask are used.

## Conditioning-debug result

Prompt 0, after correcting right padding:

### Direct answer: before versus after SANA conditioning

| Comparison point | Native vs mapped cosine |
|---|---:|
| **Before SANA's conditioning transforms** | **0.849850** |
| After the complete caption projection and RMSNorm | **0.509572** |

So the clean result is: **cosine fell from `0.850` to `0.510` before the text reached cross-attention.**

The intermediate measurements locate where this happened:

```text
input to fc1       0.849850
after fc1          0.944483
after GELU         0.585551
after fc2          0.737549
after RMSNorm      0.509572
```

This is the requested `cos(before operation)` versus `cos(after operation)` test. GELU and the final normalization expose/amplify the mismatch most strongly.

What those operations are:

- **`fc1`** is SANA's learned linear reduction from 2,304 Gemma channels to its 1,152 model channels.
- **GELU** is an element-wise nonlinear gate. Positive coordinates mostly pass; negative coordinates are suppressed. Native and mapped coordinates that fall on different sides of this gate produce the drop from `0.944` to `0.586`.
- **`fc2`** is the second learned linear layer, from 1,152 to 1,152 channels. Its increase to `0.738` does not recover lost semantics: its output norms still differ by about 59%, and a linear projection can make global directions look closer while discarding distinguishing coordinates.
- **RMSNorm** computes approximately `learned_weight * x / RMS(x)` per token. A plain scalar normalization would preserve cosine, but SANA's learned per-channel weights reweight dimensions. This reduces the cosine to `0.510` for the out-of-distribution mapped activations.

The attention-weight cosine near zero is **not** a direct `0.510 -> 0.003` linear transformation. It compares full native and mapped forward passes. At block 0, Q starts the same and attention-weight cosine is `0.524`; at later blocks Q has also changed because every earlier conditioning result altered the image state. Softmax then converts small changes in token ranking into sharply different token-selection distributions. Two nearly one-hot distributions selecting different tokens have cosine near zero. This accumulated divergence produces `0.003` at block 23.

To isolate only the immediate effect of the mapped text at every block, the next diagnostic should hold the native image query fixed and compare:

```text
softmax(Q_native K_native^T) versus softmax(Q_native K_mapped^T)
```

The existing late-block attention-weight number also changes Q, so it should not be described as the direct output of one text projection.

### What K, V, and attention weights mean

For each cross-attention block, SANA approximately computes:

```text
Q = Wq(image tokens)
K = Wk(text conditioning)
V = Wv(text conditioning)
A = softmax(Q K^T / sqrt(head_dim) + mask)
output = Wo(A V)
```

- **Q (query):** what each current image-latent token is looking for.
- **K (key):** how each text token is indexed for matching against an image query. It mainly controls *which text token is selected*.
- **V (value):** the semantic content retrieved from the selected text token. It controls *what information is injected* into the image state.
- **Attention weights (`A`):** the probability distribution over text tokens after the Q/K scores and softmax. Near-zero native/mapped weight cosine means the two runs attend to different text tokens.

There is no separately mapped K or V file. The two compared paths are:

```text
native Gemma embedding -> SANA caption projection -> K_native, V_native
mapped LLaVA embedding -> SANA caption projection -> K_mapped, V_mapped
```

Both paths use the **same frozen SANA weights**. Therefore, “K cosine” means `cos(K_native, K_mapped)`, “V cosine” means `cos(V_native, V_mapped)`, and “attention-weight cosine” compares the resulting native and mapped attention distributions. At block 0 the image query starts identically; in later blocks Q also differs because earlier mapped conditioning has already changed the image state.

| Stage | Native/mapped cosine | Key observation |
|---|---:|---|
| Raw common active tokens | 0.849850 | Inputs already differ materially |
| After `fc1` | 0.944483 | Linear projection does not expose the failure |
| After GELU | **0.585551** | First major amplification |
| After `fc2` | 0.737549 | Norms diverge strongly |
| After RMSNorm | **0.509572** | Functionally relevant representation differs greatly |
| Denoiser output at one timestep | 0.995343 | Misleadingly high because the latent/residual dominates |

Selected cross-attention results:

| Block | K cosine | V cosine | Attention-weight cosine |
|---:|---:|---:|---:|
| 0 | 0.999932 | 0.596571 | 0.523768 |
| 10 | 0.999988 | 0.491049 | 0.518030 |
| 20 | 0.999987 | 0.576824 | 0.154092 |
| 23 | 0.999935 | 0.632076 | **0.003046** |
| 24 | 0.999771 | 0.654888 | **0.016324** |
| 27 | 0.999868 | 0.745165 | 0.210957 |

The K projections remain deceptively similar, while V projections and actual attention weights diverge. The failure becomes clear after GELU/RMSNorm and inside cross-attention.

These cosine values are worse than they first appear:

- `V cosine = 0.49-0.75` is a large content error, not a close match. A cosine of `0.50` is about a 60-degree directional difference.
- Attention-weight cosine `0.003` or `0.016` is effectively no overlap: native and mapped runs retrieve different text tokens.
- K cosine near `1.0` is insufficient. Cosine ignores magnitude, attention uses Q/K dot products followed by an exponential softmax, and later-layer Q has already diverged.
- Denoiser-output cosine `0.995` at one timestep is also misleading because the large image/residual path dominates that tensor. The smaller text-conditioned change is accumulated over all 20 denoising steps.

The functionally important result is therefore not the high K or final-output cosine. It is the collapse from raw text cosine `0.850` to RMSNorm cosine `0.510`, V cosine around `0.49-0.75`, and late attention-weight cosine near zero. That is fully consistent with the catastrophic GenEval scores.

## Padding alignment issue

- The projection builder used the Gemma tokenizer's default **left padding**.
- SANA inference explicitly uses **right padding**.
- Native and mapped masks initially had no active absolute positions in common, which made the position-by-position debugger invalid.
- The builder now uses right padding and can repack an existing projection without recomputing LLaVA embeddings.
- Corrected artifact: `output/text_embeddings/sana_llava_token_aligned_geneval_native_rightpad.pt`
- Shape: `[553, 300, 2304]`; active prompt tokens: `4,251`.

This repair did **not** materially change generation because SANA applies the mask and packs only active rows before cross-attention:

```text
y = y.masked_select(mask).view(1, active_tokens, hidden_size)
```

Left-padded active rows and the same rows moved to right-padded positions therefore become the same ordered active sequence after packing. The fix was necessary to compare native and mapped tokens at corresponding absolute positions, but it was not a meaningful model-conditioning change. This is confirmed by GenEval changing only from `0.00920` to `0.00972`.

The native-reinjection sanity check now passes exactly. The remaining native/mapped mask difference is a separate one-token/BOS alignment issue; using the mapped mask with the native embedding changes the denoiser output (`mean absolute difference: 0.05875`).

## Files changed

- `scripts/inference_geneval.py` — opt-in debugger and projected-conditioning integration.
- `diffusion/utils/conditioning_debugger.py` — stage, K/V, attention-logit, attention-weight, and reinjection comparisons.
- `tools/build_sana_llava_geneval_projection.py` — right-padding fix and fast `--repack-existing` mode.

## Current conclusion

The poor score is caused by the mapped conditioning itself, not left-versus-right padding. The strongest measured failures remain the drop from `0.850` raw token cosine to `0.510` after SANA conditioning, the missing/different BOS mask position, and the subsequent divergence of V content and attention distributions.

## Extended debugger result

The opt-in debugger now also records:

- pre/post K and V projection error, including relative L2, norm ratio, MAE, and maximum error;
- top input dimensions by raw error and estimated contribution through `Wv`, with matching `Wk`/`Wv` column norms;
- fixed-native-Q attention cosine, JS divergence, argmax agreement, top-3 overlap, and entropy;
- pre-GELU sign agreement at magnitude thresholds, magnitude-weighted sign error, and the largest post-GELU coordinate errors;
- worst active tokens by post-GELU cosine, post-RMSNorm cosine, V cosine, and sign agreement.

Each debug run writes both JSON and a flattened comparison-friendly CSV. No projector or inference behavior was changed.

Prompt 0 produced the following main findings:

- Pre-GELU sign agreement is `94.60%`; magnitude-weighted sign error is only `1.81%`. Widespread sign reversal is therefore not the main failure.
- GELU still changes cosine from `0.9445` to `0.5856`. Only `0.71%` of coordinate errors grow in absolute size, so the drop is driven by sparse large mismatches and very different post-GELU feature magnitudes, not broad error amplification.
- The first token `a` is especially damaged: post-GELU cosine `0.2131`, post-RMSNorm cosine `0.0907`, and minimum V cosine `0.0230`. The later `a` token remains much better (`0.9589` post-GELU), indicating token/context alignment failure rather than a word-level failure alone.
- V relative L2 error is approximately `0.75-1.11` across blocks: the V error is about as large as the native V signal.
- Fixed-native-Q attention already collapses: block 0 cosine `0.0652`, block 22 `0.0000`, block 23 `0.00048`, and block 24 `0.0000`. JS divergence approaches its maximum `ln(2) = 0.693`, and argmax-token agreement is usually near zero.
- Therefore late attention collapse is not primarily caused by accumulated Q drift. Changing only native K to mapped K is sufficient.
- Raw K cosine near `1.0` is a bias artifact. For example, block 0 changes from `0.693` for bias-free `Wk(x)` and `0.606` for token-centered K to `0.99993` after adding a K bias with norm `110.55`. Other blocks show the same pattern: useful bias-free K is roughly `0.59-0.83`, while biased K is roughly `0.9996-0.99999`.
- Removing K bias has essentially no effect on attention: block 0 fixed-Q attention is `0.065205` with bias and `0.065221` without it, with attention MAE around `1e-5`. This confirms that the shared bias inflates K cosine but cancels from token selection.
- Repeated high-impact V-error input dimensions include `920`, `790`, `741`, `514`, `453`, and `883`. Wv strongly amplifies their input errors; Wk does not simply ignore all of them, so K/V weight selectivity alone does not explain the apparent K agreement.

The remaining mask mismatch is one missing/different BOS position (`native=6`, `mapped=5`), but the fixed-Q comparison uses the five common tokens and still collapses. BOS mismatch is an additional problem, not the primary explanation.
