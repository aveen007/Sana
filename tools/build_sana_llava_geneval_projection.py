#!/usr/bin/env python3
"""Build native SANA GenEval conditioning from a fitted LLaVA class projector.

The calculation notebook fits a kernel from LLaVA class-span input embeddings
to SANA class-span embeddings.  This tool applies that fitted kernel to every
LLaVA subword token in each official GenEval prompt, pads the resulting
sequence to SANA's native length, and writes the bundle consumed by
scripts/inference_geneval.py.

Only the Hugging Face shard containing LLaVA's input embedding table is
downloaded; the full 7B model is not loaded.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/text_embeddings/sana_llava_projected_geneval_native.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
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
    if not path.is_file():
        raise FileNotFoundError(f"Projector does not exist: {path}")
    projector = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(projector, dict) or projector.get("kind") != "sana_class_kernel_projector":
        raise ValueError(f"Not a SANA class kernel projector: {path}")

    required_tensors = {
        "X_center_mean",
        "train_features",
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

    train_rows, source_dim = projector["train_features"].shape
    target_dim = projector["direction_coefficients"].shape[1]
    if projector["direction_coefficients"].shape[0] != train_rows:
        raise ValueError("Direction coefficients do not match the training rows")
    if projector["norm_coefficients"].shape[0] != train_rows:
        raise ValueError("Norm coefficients do not match the training rows")
    if projector["norm_coefficients"].shape[1] != 1:
        raise ValueError("Norm coefficients must have one output column")
    if projector["X_center_mean"].shape != (1, source_dim):
        raise ValueError("X_center_mean does not match the source dimension")
    if projector["Y_direction_mean"].shape != (1, target_dim):
        raise ValueError("Y_direction_mean does not match the target dimension")
    if projector["Y_cholesky"].shape != (target_dim, target_dim):
        raise ValueError("Y_cholesky does not match the target dimension")
    if projector["Y_log_norm_mean"].shape != (1, 1) or projector["Y_log_norm_std"].shape != (1, 1):
        raise ValueError("Y log-norm reconstruction tensors must have shape [1, 1]")

    kernel = projector.get("kernel")
    if not isinstance(kernel, dict) or kernel.get("kind") != "polynomial":
        raise ValueError("Only the fitted polynomial projector is supported")
    return projector


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


def load_tokenizer(model_id: str, cache_dir: Path | None):
    # Transformers 4.57 may treat an absent optional chat-template directory
    # as fatal with some huggingface_hub versions. No chat template is used by
    # this text-only extraction, so disable that optional repository lookup.
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
        raise RuntimeError("A fast tokenizer is required")
    return tokenizer


def tokenize_prompts(tokenizer, prompts: list[str], sequence_length: int) -> tuple[list[list[int]], torch.Tensor]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=sequence_length,
    )
    token_rows = [list(map(int, row)) for row in encoded["input_ids"]]
    empty = [index for index, row in enumerate(token_rows) if not row]
    if empty:
        raise ValueError(f"Tokenizer produced empty rows for prompt indices: {empty[:10]}")
    attention_mask = torch.zeros(len(prompts), sequence_length, dtype=torch.uint8)
    for index, token_ids in enumerate(token_rows):
        attention_mask[index, : len(token_ids)] = 1
    return token_rows, attention_mask


@torch.inference_mode()
def project_tokens(
    token_rows: list[list[int]],
    embedding_table: torch.Tensor,
    projector: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    flat_ids = torch.tensor([token for row in token_rows for token in row], dtype=torch.long)
    if int(flat_ids.max()) >= embedding_table.shape[0]:
        raise ValueError(
            f"Tokenizer ID {int(flat_ids.max())} exceeds embedding vocabulary {embedding_table.shape[0]}"
        )

    x_center_mean = projector["X_center_mean"].float().to(device)
    train_features = projector["train_features"].float().to(device)
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

    if embedding_table.shape[1] != train_features.shape[1]:
        raise ValueError(
            f"LLaVA embedding dimension {embedding_table.shape[1]} does not match "
            f"projector source dimension {train_features.shape[1]}"
        )

    output_parts = []
    for start in tqdm(range(0, len(flat_ids), batch_size), desc="Projecting LLaVA tokens"):
        end = min(start + batch_size, len(flat_ids))
        ids = flat_ids[start:end]
        source = embedding_table.index_select(0, ids).float().to(device)
        source = F.normalize(source, dim=1, eps=1e-12)
        source = F.normalize(source - x_center_mean, dim=1, eps=1e-12)

        similarities = sigma * (source @ train_features.T) + coef0
        kernel_rows = similarities.pow(degree)
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
    return torch.cat(output_parts, dim=0)


def pack_dense_sequence(
    projected_tokens: torch.Tensor,
    token_rows: list[list[int]],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_dim = projected_tokens.shape[1]
    y_pred = torch.zeros(len(token_rows), sequence_length, target_dim, dtype=projected_tokens.dtype)
    token_ids = torch.zeros(len(token_rows), sequence_length, dtype=torch.int32)
    offset = 0
    for row_index, row in enumerate(token_rows):
        count = len(row)
        y_pred[row_index, :count].copy_(projected_tokens[offset : offset + count])
        token_ids[row_index, :count] = torch.tensor(row, dtype=torch.int32)
        offset += count
    if offset != len(projected_tokens):
        raise RuntimeError("Projected token packing did not consume every row")
    return y_pred, token_ids


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
    model_id = str(projector["source_model"])
    print(f"Projector: {args.projector}")
    print(f"Source model: {model_id}")
    print(f"Official prompts: {len(prompts)}")
    print(f"Native output shape: [{len(prompts)}, {args.sequence_length}, {projector['Y_cholesky'].shape[0]}]")
    if args.dry_run:
        return

    tokenizer = load_tokenizer(model_id, args.cache_dir)
    token_rows, attention_mask = tokenize_prompts(tokenizer, prompts, args.sequence_length)
    print(f"LLaVA tokens to project: {sum(map(len, token_rows)):,}")
    print(f"Maximum prompt tokens: {max(map(len, token_rows))}")

    print("Locating the LLaVA input embedding shard...")
    embedding_table, tensor_key, shard_name = load_llava_embedding_table(model_id, args.cache_dir)
    print(f"Embedding tensor: {tensor_key} {tuple(embedding_table.shape)}")
    print(f"Downloaded/cached shard: {shard_name}")

    projected_tokens = project_tokens(
        token_rows,
        embedding_table,
        projector,
        torch.device(args.device),
        args.batch_size,
    )
    y_pred, token_ids = pack_dense_sequence(projected_tokens, token_rows, args.sequence_length)
    bundle = {
        "format_version": 2,
        "kind": "sana_projected_official_geneval_native_sequence",
        "projected_text_mode": "native_sequence",
        "Y_pred": y_pred,
        "attention_mask": attention_mask,
        "token_ids": token_ids,
        "prompt_texts": prompts,
        "ids": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts],
        "source_model": model_id,
        "source_representation": "llava_input_embedding_token_sequence",
        "target_model": projector["target_model"],
        "target_representation": "projected_sana_token_sequence",
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
    print(f"Saved and verified: {args.output}")
    print(f"Y_pred: {tuple(y_pred.shape)} {y_pred.dtype}")
    print(f"Attention mask: {tuple(attention_mask.shape)}")


if __name__ == "__main__":
    main()
