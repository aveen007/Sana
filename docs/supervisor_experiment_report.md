# Supervisor Progress Report: Cross-Model Embedding Adaptation

**Reporting date:** 23 September 2026  
**Main current task:** replace the native text representation of an image generator with a projected representation from a vision-language model (VLM), without retraining the generator.

## 1. Executive summary

The main image-generation study uses **SANA-0.6B at 1024 px** and the standard **GenEval** benchmark. I first reproduced the native checkpoint at **0.66732 GenEval**, close to the repository's reported **0.68**. This established a valid local baseline and scoring pipeline.

I then projected contextual LLaVA text features into SANA's Gemma text-conditioning space. Several projection variants achieved high reconstruction cosine on their calculation data but failed badly during image generation. GenEval scores ranged from **0.00920 to 0.02738**, versus **0.66732** for native SANA.

To find the cause, I instrumented SANA's real conditioning path: caption projection, GELU, RMSNorm, per-block cross-attention keys and values, fixed-query attention, token importance, and denoiser output. This showed that one highly attended first token, `a`, was mapped incorrectly. Its error caused the attention distribution to select different tokens even though aggregate embedding cosine looked high.

The root cause was then traced further back to source tokenization. The original LLaVA calculation used the wrong special-token/padding behavior, so the first contextual token was learned from a different context. I regenerated the calculation data using **BOS/special tokens and right padding**, refitted the projector, and reran the debugger. The first `a` improved from RMSNorm/K/V cosines of approximately **0.091/0.113/0.232** to **1.000/1.000/1.000**. Fixed-query attention at block 23 improved from **0.00048 to 0.99938**. The specific catastrophic attention failure is therefore fixed. A full corrected GenEval run is currently in progress.

## 2. Repositories and systems investigated

| Repository/system | Work performed | Status |
|---|---|---|
| **SANA** | Native inference, complete GenEval reproduction, several projected-conditioning methods, native-sequence injection, extensive conditioning debugger, GPU-safe evaluation wrappers | Main active experiment |
| **Ovis-Image** | Installed and inspected as a higher-capacity image-generation alternative; added sequential CPU offload and resumable native GenEval generation | Functional preparation completed; no local full score recorded because of high runtime/resource cost |
| **OmniGen2** | Installed and inspected text-to-image, image-editing, in-context generation, and understanding workflows; prepared/tested example editing commands | Exploratory setup; no local full benchmark result recorded |
| **SAM-Audio** | Earlier VLM-to-T5 adaptation study using pooled, token-sequence, EOS, and position-sensitive projection variants | Completed comparative experiments; still below native T5 |
| **Microsoft CLAP + ESC-50** | Native zero-shot reproduction and Qwen3-VL-to-CLAP projection experiment | Complete local benchmark and projection result |
| **CLIPSep** | Repository setup and dataset investigation | Stopped because VGGSound/YouTube data could not be downloaded on the server or locally |

Paper/repository reference scores are included below only for context and are explicitly separated from locally reproduced results.

## 3. SANA native GenEval reproduction

### Evaluation setup

| Setting | Value |
|---|---|
| Model | SANA-0.6B, 1024 px multilingual checkpoint |
| Prompts | Official GenEval, 553 prompts |
| Images per prompt | 4 |
| Total expected images | 2,212 |
| Sampler | Flow-DPM-Solver |
| Sampling steps | 20 |
| CFG scale | 4.5 |
| Flow shift | 4.0 |
| Precision | float16 |
| Seed | 0 |
| Scoring | Official GenEval detector-based scorer |

### Native result

| Metric | Local SANA reproduction | SANA repository reference |
|---|---:|---:|
| **Overall GenEval** | **0.66732** | **0.68** |
| Correct images | 65.19% | - |
| Correct prompts | 78.30% | - |

| Task | Local score | Correct/total |
|---|---:|---:|
| Single object | 99.38% | 318/320 |
| Two objects | 78.28% | 310/396 |
| Counting | 70.00% | 224/320 |
| Colors | 87.23% | 328/376 |
| Position | 25.50% | 102/400 |
| Color attribution | 40.00% | 160/400 |

The small difference from the reported 0.68 is consistent with an ordinary reproduction difference; the local result can be reported as **0.67**.

## 4. SANA projection experiments

### GenEval summary

| Experiment | Overall | Correct images | Correct prompts | Interpretation |
|---|---:|---:|---:|---|
| Native SANA baseline | **0.66732** | 65.19% | 78.30% | Valid reference |
| Initial projected embeddings | 0.01546 | 1.45% | 3.98% | Projection reached inference but conditioning failed |
| Class/kernel projected native sequence | 0.02738 | 2.53% | 4.70% | Best projected score so far, still catastrophic |
| Token-aligned projection, original file | 0.00920 | 0.81% | 2.35% | Severe token-level failure |
| Same token projection, right-padded repack | 0.00972 | 0.86% | 2.53% | Repacking positions alone did not fix contextual features |
| **BOS + right-padding projector** | **Pending** | **Pending** | **Pending** | Attention collapse fixed in debugger; full GenEval running |

One class/kernel run contained 2,211 rather than 2,212 images, so its single-object denominator was 319 instead of 320. The scorer still completed, but this should be noted when comparing exact counts.

### Previous projected task breakdown

| Task | Native | Class/kernel | Token-aligned right-padded repack |
|---|---:|---:|---:|
| Counting | 70.00% | 0.00% | 0.00% |
| Color attribution | 40.00% | 0.00% | 0.00% |
| Colors | 87.23% | 4.26% | 0.27% |
| Position | 25.50% | 0.00% | 0.00% |
| Single object | 99.38% | 10.66% | 5.31% |
| Two objects | 78.28% | 1.52% | 0.25% |

## 5. Projection-data construction

### 5.1 Class/kernel experiment

1. Began with **3,000 WordNet noun classes**.
2. Explicitly excluded classes appearing in the GenEval benchmark to keep official evaluation concepts out of projector fitting.
3. Used **five prompt templates per class** and averaged the template representations into one class prototype.
4. Extracted LLaVA source representations with dimension **4,096** and native SANA/Gemma target representations with dimension **2,304**.
5. Used direction/norm decomposition, Cholesky target whitening, and polynomial kernel ridge regression.
6. Injected projected results back into SANA's native `[300, 2304]` conditioning layout for the 553 untouched official prompts.

Projection-side results looked extremely strong:

| Metric | Class/kernel projector |
|---|---:|
| Correct cosine | 0.999998 |
| Wrong cosine | 0.608208 |
| Separation gap | 0.391790 |
| Top-1 | 1.000 |
| Downstream GenEval | **0.02738** |

This established that in-sample reconstruction cosine alone is not a reliable measure of functional equivalence inside the generator.

### 5.2 Token-aligned experiment

To preserve the sequence rather than project one averaged class vector:

1. Used **500 benchmark-excluded classes** and one template: `a bad photo of a {}.`
2. Constructed 500 calculation prompts and **3,793 aligned target-token rows**.
3. Matched contextual LLaVA tokens to contextual Gemma tokens by character-span overlap.
4. Used a polynomial-kernel Nystrom approximation with **2,048 landmarks**.
5. For official inference, projected **4,251 prompt tokens** and packed them into SANA's native `[553, 300, 2304]` layout with an attention mask.
6. Official GenEval prompts were used only after projector fitting.

The corrected BOS/right-padding fit produced:

| Metric | Corrected token projector |
|---|---:|
| Raw correct cosine | 0.978771 |
| Raw wrong cosine | 0.591157 |
| Raw separation gap | 0.387613 |
| Raw global top-1 | 0.179014 |
| Centered correct cosine | 0.958288 |
| Centered wrong cosine | 0.005638 |
| Centered separation gap | 0.952650 |
| Centered global top-1 | 0.199842 |

These are calculation-set diagnostics, not substitutes for GenEval.

## 6. Conditioning debugger and root-cause analysis

### Instrumented SANA pathway

```text
Gemma-compatible sequence [300, 2304]
  -> caption projection fc1 [2304 -> 1152]
  -> GELU
  -> fc2 [1152 -> 1152]
  -> RMSNorm
  -> per-block cross-attention K/V projections
  -> softmax attention over text tokens
  -> image-latent conditioning
```

The debugger compares native and mapped tensors at every stage, includes a fixed-native-query attention test, and verifies that manually reinjecting the native embedding at the projected-injection point reproduces the normal native run exactly.

### Original failure

For prompt 0, the first `a` was both the most important native token and the most seriously damaged relevant token:

| First `a` diagnostic | Original projector |
|---|---:|
| Raw cosine | 0.613 |
| Post-GELU cosine | 0.213 |
| Post-RMSNorm cosine | 0.091 |
| Bias-free K cosine | 0.113 |
| V cosine | 0.232 |
| Relative L2 error | 3.47 |
| Native attention importance | 0.686 |

At block 23, its attention probability changed from approximately **0.9973 to 0.0000**. The mapped path therefore discarded the token that native SANA relied on most.

The debugger also showed why raw K cosine was misleading. A very large shared K bias raised the displayed K cosine to approximately 0.999, but the shared bias cancels in the token-wise softmax. Bias-free K and the resulting attention distribution revealed the real failure.

### Corrected BOS/right-padding result

The source-token extraction was regenerated with:

```text
add_special_tokens = True
padding_side = right
```

The projector was then refitted rather than merely repacking its final output.

| Diagnostic | Original | Corrected |
|---|---:|---:|
| First `a` raw cosine | 0.613 | **1.000** |
| First `a` post-GELU cosine | 0.213 | **1.000** |
| First `a` post-RMSNorm cosine | 0.091 | **1.000** |
| First `a` bias-free K cosine | 0.113 | **1.000** |
| First `a` V cosine | 0.232 | **1.000** |
| Block 0 fixed-Q attention cosine | 0.065 | **0.964** |
| Block 22 fixed-Q attention cosine | approximately 0.000 | **0.981** |
| Block 23 fixed-Q attention cosine | 0.00048 | **0.99938** |
| Block 24 fixed-Q attention cosine | approximately 0.000 | **0.988** |
| Denoiser-output cosine at debug timestep | 0.99534 | **0.99818** |

The native-reinjection sanity check passes with exactly zero error. The specific first-token/attention-collapse mechanism is therefore fixed.

### Remaining limitations

- Native and mapped masks still contain 6 versus 5 active positions. This is now a separate, known issue rather than the cause of the original first-token failure.
- `bench` remains poorly reconstructed (post-RMSNorm cosine 0.345 and V cosine 0.304), but its native attention importance is only 0.017.
- `photo` and `of` remain imperfect, also with low native attention importance.
- The corrected projector must still be judged by the complete 20-step, 2,212-image GenEval result.

## 7. Engineering work completed for SANA

- Adapted the repository to load Hugging Face checkpoint paths directly.
- Reproduced GenEval generation and detector-based scoring in the existing environment.
- Built `mmcv-full` with CUDA operations and installed compatible MMDetection/CLIP scoring dependencies without replacing the existing PyTorch installation.
- Added native-sequence projected-conditioning loading to inference.
- Added class-level and contextual token-level projector builders.
- Added exact prompt/order/hash checks to prevent silent benchmark misalignment.
- Added resumable/cached artifact handling and unique run labels to prevent reuse of previous images.
- Added an opt-in conditioning debugger with JSON and CSV reports.
- Added physical-GPU selection that respects `CUDA_VISIBLE_DEVICES`; current runs use physical GPU 1.
- Added debug-only and full-evaluation wrappers for the corrected projector.

## 8. Other image-generation repositories

### Ovis-Image

Ovis-Image was cloned and investigated before selecting SANA for the main iterative work. Its repository reports **0.84 GenEval** for Ovis-Image, but this is an external reference and was not reproduced locally.

Local work completed:

- implemented sequential CPU offload so inference can run on a 24 GB GPU;
- implemented native GenEval generation using the official 553-prompt layout;
- made GenEval generation resumable so a long run can continue from already generated images.

A complete local GenEval score is not recorded. Full generation was judged too expensive for rapid projection/debug iterations because expected runtime was in the tens of hours.

### OmniGen2

OmniGen2 was cloned and its main modes were inspected/tested:

- text-to-image generation;
- instruction-based image editing;
- in-context generation;
- multimodal understanding/chat;
- model, sequential, and group CPU-offload paths;
- example editing runs with different step counts and guidance settings.

The repository reports **0.80 GenEval** for OmniGen2; this is a paper/repository number, not a local reproduction. No completed local OmniGen2 benchmark output is present in the workspace.

### Why SANA became the main testbed

SANA provides a smaller 600M diffusion transformer, 20-step image generation, an available 1024 px checkpoint, and repository-integrated GenEval scripts. This made it more practical for repeated full-benchmark and internal-conditioning experiments than the larger Ovis-Image and OmniGen2 pipelines.

## 9. Related non-image experiments

### SAM-Audio VLM-to-T5 adaptation

This was the earlier cross-model adaptation testbed and motivated the SANA work.

| Experiment | JudgeOverall | CLAP similarity |
|---|---:|---:|
| Native T5 | **4.492** | **0.257** |
| Pooled single-token projection | 1.268 | 0.003 |
| Token-sequence projection | 2.494 | 0.033 |
| Token sequence + native EOS | 2.117 | 0.061 |
| Fourier subspan projection | 2.503 | 0.032 |

The token-sequence version improved substantially over one pooled token but remained below native T5. The Fourier subspan experiment resolved all 1,310 previously collapsed positional pairs mechanically, but downstream quality remained effectively unchanged. This was an early indication that good reconstruction diagnostics do not guarantee downstream equivalence.

### Microsoft CLAP and ESC-50

- Reproduced zero-shot CLAP on all 2,000 ESC-50 clips.
- Local accuracy: **94.35% (1,887/2,000)**.
- Repository/paper zero-shot reference: **93.90%**.
- Runtime: approximately **31 seconds** on GPU.
- Then projected Qwen3-VL features into CLAP's text space using 5,864 fit prompts, 198 validation prompts, and 50 leakage-controlled ESC-50 class prompts.
- Projection validation reached **0.926 paired cosine** and **89.4% retrieval accuracy**, but final projected ESC-50 accuracy was only **37.3%**.
- Passing native CLAP encoder features through the same bypass evaluator recovered **94.35%**, validating the evaluation/injection path and locating the failure in projection generalization.

### CLIPSep

CLIPSep was cloned and set up, but its main evaluation/training path required VGGSound media hosted on YouTube. Downloads failed on both the server and local machine because the required content could not be accessed through the available network/proxy setup, so this direction was stopped.

## 10. Current status and next decision

1. The corrected SANA BOS/right-padding projector has passed the one-prompt internal attention test.
2. The corrected 553-prompt conditioning bundle has been built successfully.
3. Full GenEval generation/scoring is running on physical GPU 1.
4. When it completes, add the overall score, six task scores, correct-image percentage, and correct-prompt percentage to the pending row in Section 4.
5. If the score improves materially, test more prompts with the debugger and then address the remaining mask mismatch. If it remains near zero, compare full denoising trajectories rather than only one debug timestep.

## 11. Main scientific conclusion so far

Across SAM-Audio, CLAP, and SANA, high cosine similarity in the fitted embedding space has repeatedly failed to guarantee downstream task preservation. The SANA debugger provides the clearest explanation: a small number of context-sensitive token errors can occur in dimensions strongly used by K/V projections, and softmax attention can amplify those errors into a different token-selection decision. Correct tokenizer context is therefore part of the representation being projected; matching only vector dimensions or aggregate cosine is insufficient.
