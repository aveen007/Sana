#!/usr/bin/env python3
"""Build native SANA GenEval conditioning from a fitted LLaVA projector.

Two projector formats are supported:

* ``sana_token_aligned_nystrom_projector`` encodes each official prompt with
  contextual LLaVA states, aligns those states to SANA/Gemma token spans by
  character overlap, projects the aligned states, and places them in the exact
  native 300-position SANA layout.
* ``sana_class_kernel_projector`` retains the legacy input-embedding lookup
  path for previously produced artifacts.

The official GenEval prompts are used only to build inference conditioning;
they are never added to the projector fitting data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_METADATA = Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl")
LEGACY_PROJECTOR_KIND = "sana_class_kernel_projector"
TOKEN_ALIGNED_PROJECTOR_KIND = "sana_token_aligned_nystrom_projector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument(
        "--source-token-embeddings",
        type=Path,
        help=(
            "Optional compact legacy LLaVA token bundle. This is supported "
            "only by sana_class_kernel_projector artifacts."
        ),
    )
    parser.add_argument("--eval-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/text_embeddings/sana_llava_projected_geneval_native.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of aligned token rows projected per kernel batch.",
    )
    parser.add_argument(
        "--source-batch-size",
        type=int,
        default=2,
        help="Number of GenEval prompts passed through contextual LLaVA at once.",
    )
    parser.add_argument("--source-max-length", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "source_batch_size", "source_max_length", "sequence_length"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_projector(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Projector does not exist: {path}")
    projector = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(projector, dict):
        raise TypeError(f"Projector must be a dictionary: {path}")

    kind = projector.get("kind")
    if kind == LEGACY_PROJECTOR_KIND:
        basis_name = "train_features"
    elif kind == TOKEN_ALIGNED_PROJECTOR_KIND:
        basis_name = "landmark_features"
    else:
        raise ValueError(f"Unsupported projector kind {kind!r}: {path}")

    required_tensors = {
        "X_center_mean",
        basis_name,
        "direction_coefficients",
        "norm_coefficients",
        "Y_direction_mean",
        "Y_cholesky",
        "Y_log_norm_mean",
        "Y_log_norm_std",
    }
    for name in required_tensors:
        tensor = projector.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"Projector tensor {name!r} has an invalid shape")

    basis_rows, source_dim = projector[basis_name].shape
    target_dim = projector["direction_coefficients"].shape[1]
    if projector["direction_coefficients"].shape[0] != basis_rows:
        raise ValueError("Direction coefficients do not match the kernel basis")
    if projector["norm_coefficients"].shape != (basis_rows, 1):
        raise ValueError("Norm coefficients do not match the kernel basis")
    if projector["X_center_mean"].shape != (1, source_dim):
        raise ValueError("X_center_mean does not match the source dimension")
    if projector["Y_direction_mean"].shape != (1, target_dim):
        raise ValueError("Y_direction_mean does not match the target dimension")
    if projector["Y_cholesky"].shape != (target_dim, target_dim):
        raise ValueError("Y_cholesky does not match the target dimension")
    if projector["Y_log_norm_mean"].shape != (1, 1):
        raise ValueError("Y_log_norm_mean must have shape [1, 1]")
    if projector["Y_log_norm_std"].shape != (1, 1):
        raise ValueError("Y_log_norm_std must have shape [1, 1]")
    if kind == TOKEN_ALIGNED_PROJECTOR_KIND:
        if int(projector.get("source_hidden_dim", source_dim)) != source_dim:
            raise ValueError("source_hidden_dim does not match the landmark features")
        if int(projector.get("target_hidden_dim", target_dim)) != target_dim:
            raise ValueError("target_hidden_dim does not match the direction coefficients")
        if not isinstance(projector.get("chi_prompt"), str):
            raise ValueError("Token-aligned projector has no CHI prompt string")

    kernel = projector.get("kernel")
    if not isinstance(kernel, dict) or kernel.get("kind") != "polynomial":
        raise ValueError("Only polynomial kernel projectors are supported")
    return projector


def projector_basis(projector: dict[str, Any]) -> torch.Tensor:
    if projector["kind"] == TOKEN_ALIGNED_PROJECTOR_KIND:
        return projector["landmark_features"]
    return projector["train_features"]


def load_tokenizer(model_id: str, cache_dir: Path | None):
    # Transformers 4.57 may treat an absent optional chat-template directory
    # as fatal with some huggingface_hub versions. No chat template is used.
    try:
        import transformers.tokenization_utils_base as tokenizer_base

        tokenizer_base.list_repo_templates = lambda *args, **kwargs: []
    except (AttributeError, ImportError):
        pass
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"A fast tokenizer is required for {model_id}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def overlap_weights(
    target_spans: list[tuple[int, int]],
    source_spans: list[tuple[int, int]],
) -> tuple[torch.Tensor, int]:
    if not target_spans or not source_spans:
        raise ValueError("Token alignment requires non-empty target and source spans")
    weights = torch.zeros(len(target_spans), len(source_spans), dtype=torch.float32)
    source_centers = torch.tensor(
        [(start + end) / 2 for start, end in source_spans], dtype=torch.float32
    )
    fallback_count = 0
    for target_index, (target_start, target_end) in enumerate(target_spans):
        for source_index, (source_start, source_end) in enumerate(source_spans):
            weights[target_index, source_index] = max(
                0,
                min(target_end, source_end) - max(target_start, source_start),
            )
        total = weights[target_index].sum()
        if total > 0:
            weights[target_index] /= total
        else:
            fallback_count += 1
            target_center = (target_start + target_end) / 2
            nearest = int(torch.argmin(torch.abs(source_centers - target_center)).item())
            weights[target_index, nearest] = 1.0
    return weights, fallback_count


def build_target_layouts(
    tokenizer,
    prompts: list[str],
    chi_prompt: str,
    sequence_length: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    conditioned = [chi_prompt + prompt for prompt in prompts]
    tokenizer_max_length = len(tokenizer.encode(chi_prompt)) + sequence_length - 2
    encoded = tokenizer(
        conditioned,
        max_length=tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    select_index = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.arange(tokenizer_max_length - sequence_length + 1, tokenizer_max_length),
        )
    )
    input_ids = encoded["input_ids"].index_select(1, select_index)
    attention_mask = encoded["attention_mask"].index_select(1, select_index).bool()
    special_mask = encoded["special_tokens_mask"].index_select(1, select_index).bool()
    offsets = encoded["offset_mapping"].index_select(1, select_index)

    layouts = []
    prompt_start = len(chi_prompt)
    for row_index, prompt in enumerate(prompts):
        prompt_end = prompt_start + len(prompt)
        positions = []
        spans = []
        for position in range(sequence_length):
            if not bool(attention_mask[row_index, position]):
                continue
            if bool(special_mask[row_index, position]):
                continue
            token_start = int(offsets[row_index, position, 0])
            token_end = int(offsets[row_index, position, 1])
            if token_end <= prompt_start or token_start >= prompt_end or token_end <= token_start:
                continue
            raw_start = max(0, token_start - prompt_start)
            raw_end = min(len(prompt), token_end - prompt_start)
            if raw_end <= raw_start:
                continue
            positions.append(position)
            spans.append((raw_start, raw_end))
        if not positions:
            raise ValueError(f"No native SANA prompt tokens found for {prompt!r}")
        layouts.append({"positions": positions, "spans": spans})
    return layouts, input_ids.to(dtype=torch.int32)


def load_contextual_llava(
    model_id: str,
    cache_dir: Path | None,
    device: torch.device,
):
    from transformers import LlavaForConditionalGeneration

    tokenizer = load_tokenizer(model_id, cache_dir)
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    model.requires_grad_(False)
    if device.type != "cuda":
        model = model.to(device)

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        language_model = model.model.language_model
    backbone = getattr(language_model, "model", language_model)
    input_device = model.get_input_embeddings().weight.device
    return model, backbone, tokenizer, input_device


@torch.inference_mode()
def encode_and_align_contextual_llava(
    prompts: list[str],
    target_layouts: list[dict[str, Any]],
    model,
    backbone,
    tokenizer,
    input_device: torch.device,
    source_batch_size: int,
    source_max_length: int,
) -> tuple[torch.Tensor, int]:
    # Keep an explicit model argument so Accelerate's dispatch hooks remain
    # alive for the duration of the backbone-only forward passes.
    _ = model
    aligned_parts = []
    total_fallbacks = 0
    for start in tqdm(
        range(0, len(prompts), source_batch_size),
        desc="Contextual LLaVA alignment",
    ):
        end = min(start + source_batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        encoded = tokenizer(
            batch_prompts,
            # The calculation notebook used the normal LLaVA BOS token. Keep
            # it here so contextual states match the fitted projector domain.
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=source_max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        attention_mask_cpu = encoded["attention_mask"].bool()
        outputs = backbone(
            input_ids=encoded["input_ids"].to(input_device),
            attention_mask=encoded["attention_mask"].to(input_device),
            use_cache=False,
            return_dict=True,
        )
        contextual = getattr(outputs, "last_hidden_state", outputs[0])

        for local_index, prompt in enumerate(batch_prompts):
            source_indices = []
            source_spans = []
            for token_index in range(offsets.shape[1]):
                if not bool(attention_mask_cpu[local_index, token_index]):
                    continue
                source_start = int(offsets[local_index, token_index, 0])
                source_end = int(offsets[local_index, token_index, 1])
                if source_end <= source_start:
                    continue
                source_indices.append(token_index)
                source_spans.append((source_start, source_end))
            if not source_indices:
                raise ValueError(f"LLaVA produced no source tokens for {prompt!r}")

            layout = target_layouts[start + local_index]
            weights, fallback_count = overlap_weights(layout["spans"], source_spans)
            total_fallbacks += fallback_count
            source_features = contextual[local_index, source_indices].float()
            aligned = weights.to(source_features.device) @ source_features
            aligned_parts.append(aligned.to(device="cpu", dtype=torch.float16))

        del outputs, contextual
    if not aligned_parts:
        raise RuntimeError("No contextual LLaVA rows were produced")
    return torch.cat(aligned_parts), total_fallbacks


@torch.inference_mode()
def project_source_features(
    source_features: torch.Tensor,
    projector: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    x_center_mean = projector["X_center_mean"].float().to(device)
    basis = projector_basis(projector).float().to(device)
    direction_coefficients = projector["direction_coefficients"].float().to(device)
    norm_coefficients = projector["norm_coefficients"].float().to(device)
    y_direction_mean = projector["Y_direction_mean"].float().to(device)
    y_cholesky = projector["Y_cholesky"].float().to(device)
    y_log_norm_mean = projector["Y_log_norm_mean"].float().to(device)
    y_log_norm_std = projector["Y_log_norm_std"].float().to(device)
    y_norm_min = float(projector["Y_norm_min"])
    y_norm_max = float(projector["Y_norm_max"])
    kernel = projector["kernel"]
    sigma = float(kernel["sigma"])
    degree = int(kernel["degree"])
    coef0 = float(kernel["coef0"])

    if source_features.ndim != 2 or source_features.shape[1] != basis.shape[1]:
        raise ValueError(
            f"Source features must have shape [rows, {basis.shape[1]}], "
            f"got {tuple(source_features.shape)}"
        )

    output_parts = []
    for start in tqdm(range(0, len(source_features), batch_size), desc="Projecting aligned tokens"):
        end = min(start + batch_size, len(source_features))
        source = source_features[start:end].float().to(device)
        source = F.normalize(source, dim=1, eps=1e-12)
        source = F.normalize(source - x_center_mean, dim=1, eps=1e-12)
        kernel_rows = (sigma * (source @ basis.T) + coef0).pow(degree)
        predicted_white = kernel_rows @ direction_coefficients
        predicted_direction = F.normalize(
            predicted_white @ y_cholesky.T + y_direction_mean,
            dim=1,
            eps=1e-12,
        )
        predicted_norm_z = kernel_rows @ norm_coefficients
        predicted_norm = (predicted_norm_z * y_log_norm_std + y_log_norm_mean).exp()
        predicted_norm = predicted_norm.clamp(min=y_norm_min, max=y_norm_max)
        output_parts.append((predicted_direction * predicted_norm).to(dtype=torch.float16).cpu())
    return torch.cat(output_parts)


def pack_target_aligned_sequence(
    projected_tokens: torch.Tensor,
    target_layouts: list[dict[str, Any]],
    target_token_ids: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_dim = projected_tokens.shape[1]
    y_pred = torch.zeros(len(target_layouts), sequence_length, target_dim, dtype=projected_tokens.dtype)
    attention_mask = torch.zeros(len(target_layouts), sequence_length, dtype=torch.uint8)
    saved_token_ids = torch.zeros(len(target_layouts), sequence_length, dtype=torch.int32)
    offset = 0
    for row_index, layout in enumerate(target_layouts):
        positions = layout["positions"]
        count = len(positions)
        position_tensor = torch.tensor(positions, dtype=torch.long)
        y_pred[row_index].index_copy_(0, position_tensor, projected_tokens[offset : offset + count])
        attention_mask[row_index, position_tensor] = 1
        saved_token_ids[row_index, position_tensor] = target_token_ids[
            row_index, position_tensor
        ].to(dtype=saved_token_ids.dtype)
        offset += count
    if offset != len(projected_tokens):
        raise RuntimeError("Native packing did not consume every projected token")
    return y_pred, attention_mask, saved_token_ids


def load_llava_embedding_table(model_id: str, cache_dir: Path | None) -> tuple[torch.Tensor, str, str]:
    cache = str(cache_dir) if cache_dir else None
    index_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json",
        cache_dir=cache,
    )
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid Safetensors index: {index_path}")
    candidates = [
        key
        for key in weight_map
        if key.endswith("language_model.model.embed_tokens.weight")
        or key.endswith("model.embed_tokens.weight")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one LLaVA input embedding tensor, found: {candidates}")
    tensor_key = candidates[0]
    shard_name = weight_map[tensor_key]
    shard_path = hf_hub_download(
        repo_id=model_id,
        filename=shard_name,
        cache_dir=cache,
    )
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        embedding_table = handle.get_tensor(tensor_key).clone()
    return embedding_table, tensor_key, shard_name


def tokenize_legacy_prompts(tokenizer, prompts: list[str], sequence_length: int):
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=sequence_length,
    )
    token_rows = [list(map(int, row)) for row in encoded["input_ids"]]
    if any(not row for row in token_rows):
        raise ValueError("The source tokenizer produced an empty GenEval prompt")
    attention_mask = torch.zeros(len(prompts), sequence_length, dtype=torch.uint8)
    for index, row in enumerate(token_rows):
        attention_mask[index, : len(row)] = 1
    return token_rows, attention_mask


def load_source_token_embeddings(
    path: Path,
    prompts: list[str],
    expected_model_id: str,
    sequence_length: int,
    expected_source_dim: int,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Source token embedding file does not exist: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("kind") != "llava_geneval_token_embeddings":
        raise ValueError(f"Not a LLaVA GenEval token embedding bundle: {path}")
    if bundle.get("model_id") != expected_model_id:
        raise ValueError("Source token model does not match the projector")
    if list(bundle.get("prompt_texts", [])) != prompts:
        raise ValueError("Source token prompts do not match official GenEval order")
    embeddings = bundle.get("embeddings")
    row_offsets = bundle.get("row_offsets")
    flat_token_ids = bundle.get("token_ids")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Source embeddings must have shape [token_count, source_dim]")
    if embeddings.shape[1] != expected_source_dim:
        raise ValueError("Source embedding dimension does not match the projector")
    if not isinstance(row_offsets, torch.Tensor) or len(row_offsets) != len(prompts) + 1:
        raise ValueError("Source row offsets do not match the prompt count")
    if not isinstance(flat_token_ids, torch.Tensor) or len(flat_token_ids) != len(embeddings):
        raise ValueError("Source token IDs do not match the embeddings")
    token_rows = []
    attention_mask = torch.zeros(len(prompts), sequence_length, dtype=torch.uint8)
    for index in range(len(prompts)):
        start = int(row_offsets[index])
        end = int(row_offsets[index + 1])
        count = end - start
        if count <= 0 or count > sequence_length:
            raise ValueError(f"Invalid source token count {count} for prompt {index}")
        token_rows.append(flat_token_ids[start:end].long().tolist())
        attention_mask[index, :count] = 1
    return token_rows, attention_mask, embeddings.contiguous()


def pack_legacy_sequence(
    projected_tokens: torch.Tensor,
    token_rows: list[list[int]],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_dim = projected_tokens.shape[1]
    y_pred = torch.zeros(len(token_rows), sequence_length, target_dim, dtype=projected_tokens.dtype)
    token_ids = torch.zeros(len(token_rows), sequence_length, dtype=torch.int32)
    attention_mask = torch.zeros(len(token_rows), sequence_length, dtype=torch.uint8)
    offset = 0
    for row_index, row in enumerate(token_rows):
        count = len(row)
        y_pred[row_index, :count].copy_(projected_tokens[offset : offset + count])
        token_ids[row_index, :count] = torch.tensor(row, dtype=torch.int32)
        attention_mask[row_index, :count] = 1
        offset += count
    if offset != len(projected_tokens):
        raise RuntimeError("Legacy packing did not consume every projected token")
    return y_pred, attention_mask, token_ids


def atomic_torch_save(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(bundle, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; pass --force to replace it")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    projector = load_projector(args.projector)
    rows = read_jsonl(args.eval_metadata)
    prompts = [row["prompt"] for row in rows]
    source_model_id = str(projector["source_model"])
    target_dim = projector["direction_coefficients"].shape[1]
    print(f"Projector: {args.projector}")
    print(f"Projector kind: {projector['kind']}")
    print(f"Source model: {source_model_id}")
    print(f"Official prompts: {len(prompts)}")
    print(f"Native output shape: [{len(prompts)}, {args.sequence_length}, {target_dim}]")
    if args.dry_run:
        return

    device = torch.device(args.device)
    fallback_count = 0
    if projector["kind"] == TOKEN_ALIGNED_PROJECTOR_KIND:
        if args.source_token_embeddings:
            raise ValueError(
                "--source-token-embeddings cannot supply the offsets/contextual states "
                "required by a token-aligned projector"
            )
        expected_length = int(projector.get("model_max_length", args.sequence_length))
        if args.sequence_length != expected_length:
            raise ValueError(
                f"Projector requires sequence length {expected_length}, got {args.sequence_length}"
            )

        target_tokenizer = load_tokenizer(str(projector["target_model"]), args.cache_dir)
        target_layouts, target_token_ids = build_target_layouts(
            target_tokenizer,
            prompts,
            projector["chi_prompt"],
            args.sequence_length,
        )
        print(f"Target-aligned prompt tokens: {sum(len(layout['positions']) for layout in target_layouts):,}")

        model, backbone, source_tokenizer, input_device = load_contextual_llava(
            source_model_id,
            args.cache_dir,
            device,
        )
        aligned_source, fallback_count = encode_and_align_contextual_llava(
            prompts,
            target_layouts,
            model,
            backbone,
            source_tokenizer,
            input_device,
            args.source_batch_size,
            args.source_max_length,
        )
        print(f"Aligned contextual source rows: {len(aligned_source):,}")
        print(f"Nearest-token alignment fallbacks: {fallback_count}")
        del backbone, model, source_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        projected_tokens = project_source_features(aligned_source, projector, device, args.batch_size)
        y_pred, attention_mask, token_ids = pack_target_aligned_sequence(
            projected_tokens,
            target_layouts,
            target_token_ids,
            args.sequence_length,
        )
        source_representation = "llava_contextual_states_aligned_to_gemma_token_spans"
        conditioning_layout = "native_gemma_prompt_positions"
    else:
        source_tokenizer = load_tokenizer(source_model_id, args.cache_dir)
        if args.source_token_embeddings:
            token_rows, attention_mask, source_features = load_source_token_embeddings(
                args.source_token_embeddings,
                prompts,
                source_model_id,
                args.sequence_length,
                projector_basis(projector).shape[1],
            )
        else:
            token_rows, attention_mask = tokenize_legacy_prompts(
                source_tokenizer, prompts, args.sequence_length
            )
            print("Locating the LLaVA input embedding shard...")
            embedding_table, tensor_key, shard_name = load_llava_embedding_table(
                source_model_id, args.cache_dir
            )
            print(f"Embedding tensor: {tensor_key} {tuple(embedding_table.shape)}")
            print(f"Downloaded/cached shard: {shard_name}")
            flat_ids = torch.tensor([token for row in token_rows for token in row], dtype=torch.long)
            if int(flat_ids.max()) >= embedding_table.shape[0]:
                raise ValueError("A source token ID exceeds the LLaVA embedding vocabulary")
            source_features = embedding_table.index_select(0, flat_ids)
        projected_tokens = project_source_features(source_features, projector, device, args.batch_size)
        y_pred, attention_mask, token_ids = pack_legacy_sequence(
            projected_tokens, token_rows, args.sequence_length
        )
        source_representation = "llava_input_embedding_token_sequence"
        conditioning_layout = "legacy_left_packed_source_tokens"

    bundle = {
        "format_version": 3,
        "kind": "sana_projected_official_geneval_native_sequence",
        "projected_text_mode": "native_sequence",
        "Y_pred": y_pred,
        "attention_mask": attention_mask,
        "token_ids": token_ids,
        "prompt_texts": prompts,
        "ids": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts],
        "source_model": source_model_id,
        "source_representation": source_representation,
        "target_model": projector["target_model"],
        "target_representation": "projected_sana_token_sequence",
        "conditioning_layout": conditioning_layout,
        "alignment_method": projector.get("alignment_method"),
        "alignment_fallback_count": fallback_count,
        "projector_kind": projector["kind"],
        "projector_sha256": sha256_file(args.projector),
        "projector_training_metrics": projector.get("training_metrics"),
        "sequence_length": args.sequence_length,
    }
    atomic_torch_save(args.output, bundle)
    saved = torch.load(args.output, map_location="cpu", weights_only=True, mmap=True)
    if saved["Y_pred"].shape != y_pred.shape or saved["attention_mask"].shape != attention_mask.shape:
        raise RuntimeError("Saved projected bundle failed shape verification")
    if not bool(torch.isfinite(saved["Y_pred"]).all()):
        raise RuntimeError("Saved projected bundle contains non-finite values")
    if int(saved["attention_mask"].sum()) != len(projected_tokens):
        raise RuntimeError("Saved attention mask does not cover every projected token")
    print(f"Saved and verified: {args.output}")
    print(f"Y_pred: {tuple(y_pred.shape)} {y_pred.dtype}")
    print(f"Attention mask: {tuple(attention_mask.shape)} ({int(attention_mask.sum())} active tokens)")


if __name__ == "__main__":
    main()
