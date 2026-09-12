#!/usr/bin/env python3
"""Export the text features consumed by SANA for paired projection training.

The script deliberately builds projection-fit prompts from GenEval's generated
training metadata and keeps the official 553-prompt suite separate. It can
export compact pooled features, the complete 300-token conditioning sequence,
or both. Outputs are sharded Safetensors files accompanied by JSONL prompt
manifests so another environment can encode exactly the same prompt ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXT_ENCODER_IDS = {
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}

OUTPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sana_config/1024ms/Sana_600M_img1024.yaml"),
        help="SANA YAML config whose text-encoder settings should be reproduced.",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("diffusion/post_training/dataset/geneval/train_metadata.jsonl"),
        help="JSONL source for projection-fit and validation prompts.",
    )
    parser.add_argument(
        "--eval-metadata",
        type=Path,
        default=Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl"),
        help="The official GenEval JSONL; these prompts are never used for fitting.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fit-prompts",
        type=int,
        default=10_000,
        help="Number of unique fit prompts. Use 0 for every prompt left after validation.",
    )
    parser.add_argument(
        "--validation-prompts",
        type=int,
        default=1_000,
        help="Number of unique training-source prompts reserved for validation.",
    )
    parser.add_argument(
        "--include-official-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export the official 553 prompts as a strictly evaluation-only split.",
    )
    parser.add_argument(
        "--representation",
        choices=("pooled", "tokens", "both"),
        default="pooled",
        help="Save compact pooled vectors, exact 300-token features, or both.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic prompt selection.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dtype",
        choices=tuple(OUTPUT_DTYPES),
        default="float16",
        help="Storage dtype for exported embeddings. Model inference remains bfloat16.",
    )
    parser.add_argument(
        "--model-id",
        help="Optional Hugging Face model override. By default it is resolved from the YAML config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted matching export and skip validated completed shards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the split without loading the encoder or writing output.",
    )
    args = parser.parse_args()

    for name in ("fit_prompts", "validation_prompts"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("batch_size", "shard_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt string in {path}:{line_number}")
            record = dict(record)
            record["prompt"] = prompt.strip()
            record["source_line"] = line_number
            records.append(record)
    return records


def unique_by_prompt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for record in records:
        prompt = record["prompt"]
        if prompt not in seen:
            unique.append(record)
            seen.add(prompt)
    return unique


def prompt_order_key(prompt: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{prompt}".encode("utf-8")).digest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_splits(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    fit_count: int,
    validation_count: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    train_unique = unique_by_prompt(train_records)
    eval_unique = unique_by_prompt(eval_records)
    eval_prompts = {record["prompt"] for record in eval_unique}

    overlap_count = sum(record["prompt"] in eval_prompts for record in train_unique)
    clean_train = [record for record in train_unique if record["prompt"] not in eval_prompts]
    clean_train.sort(key=lambda record: prompt_order_key(record["prompt"], seed))

    if validation_count >= len(clean_train):
        raise ValueError(
            f"Requested {validation_count} validation prompts, but only {len(clean_train)} non-eval prompts exist"
        )
    validation = clean_train[:validation_count]
    fit_candidates = clean_train[validation_count:]
    if fit_count:
        if fit_count > len(fit_candidates):
            raise ValueError(f"Requested {fit_count} fit prompts, but only {len(fit_candidates)} are available")
        fit = fit_candidates[:fit_count]
    else:
        fit = fit_candidates

    stats = {
        "train_rows": len(train_records),
        "train_unique": len(train_unique),
        "eval_rows": len(eval_records),
        "eval_unique": len(eval_unique),
        "excluded_train_eval_overlap": overlap_count,
    }
    return {"fit": fit, "validation": validation, "official_eval": eval_unique}, stats


def load_text_settings(config_path: Path, model_override: str | None) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    text_config = config.get("text_encoder", {})
    encoder_name = text_config.get("text_encoder_name")
    model_id = model_override or TEXT_ENCODER_IDS.get(encoder_name)
    if not model_id:
        supported = ", ".join(sorted(TEXT_ENCODER_IDS))
        raise ValueError(
            f"Cannot resolve text encoder {encoder_name!r}. Pass --model-id or use one of: {supported}"
        )

    model_max_length = int(text_config.get("model_max_length", 300))
    chi_lines = text_config.get("chi_prompt") or []
    if not isinstance(chi_lines, list) or not all(isinstance(line, str) for line in chi_lines):
        raise ValueError("text_encoder.chi_prompt must be a list of strings")
    return {
        "encoder_name": encoder_name,
        "model_id": model_id,
        "model_max_length": model_max_length,
        "chi_prompt": "\n".join(chi_lines),
    }


def split_summary(splits: dict[str, list[dict[str, Any]]], stats: dict[str, int]) -> None:
    print(
        f"Training source: {stats['train_rows']:,} rows, {stats['train_unique']:,} unique prompts; "
        f"excluded official overlap: {stats['excluded_train_eval_overlap']:,}"
    )
    print(f"Official source: {stats['eval_rows']:,} rows, {stats['eval_unique']:,} unique prompts")
    print(
        "Export splits: "
        + ", ".join(f"{name}={len(records):,}" for name, records in splits.items() if records)
    )


def prompt_manifest_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
    result = {
        "index": index,
        "prompt": record["prompt"],
        "prompt_sha256": hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
        "source_line": record["source_line"],
    }
    if "tag" in record:
        result["tag"] = record["tag"]
    return result


def write_prompt_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(prompt_manifest_record(index, record), ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def selected_conditioning(
    prompts: list[str],
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chi_prompt = settings["chi_prompt"]
    conditioned_prompts = [chi_prompt + prompt for prompt in prompts] if chi_prompt else prompts
    model_max_length = settings["model_max_length"]
    if chi_prompt:
        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
        tokenizer_max_length = num_chi_prompt_tokens + model_max_length - 2
    else:
        tokenizer_max_length = model_max_length

    tokens = tokenizer(
        conditioned_prompts,
        max_length=tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    select_index = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(
                tokenizer_max_length - model_max_length + 1,
                tokenizer_max_length,
                dtype=torch.long,
                device=device,
            ),
        )
    )
    hidden_states = text_encoder(
        tokens.input_ids,
        attention_mask=tokens.attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state.index_select(1, select_index)
    input_ids = tokens.input_ids.index_select(1, select_index)
    attention_mask = tokens.attention_mask.index_select(1, select_index)
    return hidden_states, input_ids, attention_mask


def build_payload(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    row_indices: torch.Tensor,
    representation: str,
    output_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    payload = {
        "row_indices": row_indices.to(device="cpu", dtype=torch.int32).contiguous(),
        "input_ids": input_ids.to(device="cpu", dtype=torch.int32).contiguous(),
        "attention_mask": attention_mask.to(device="cpu", dtype=torch.uint8).contiguous(),
    }

    if representation in {"pooled", "both"}:
        float_hidden = hidden_states.float()
        float_mask = attention_mask.unsqueeze(-1).float()
        pooled_mean = (float_hidden * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)
        final_token_index = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        pooled_last = hidden_states[batch_index, final_token_index]
        payload["pooled_mean"] = pooled_mean.to(device="cpu", dtype=output_dtype).contiguous()
        payload["pooled_last"] = pooled_last.to(device="cpu", dtype=output_dtype).contiguous()

    if representation in {"tokens", "both"}:
        payload["hidden_states"] = hidden_states.to(device="cpu", dtype=output_dtype).contiguous()
    return payload


def valid_completed_shard(path: Path, expected_rows: int, representation: str) -> bool:
    if not path.exists():
        return False
    required = {"row_indices", "input_ids", "attention_mask"}
    if representation in {"pooled", "both"}:
        required.update(("pooled_mean", "pooled_last"))
    if representation in {"tokens", "both"}:
        required.add("hidden_states")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if not required.issubset(handle.keys()):
                return False
            return handle.get_tensor("row_indices").shape[0] == expected_rows
    except Exception:
        return False


def export_null_conditioning(
    output_dir: Path,
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    args: argparse.Namespace,
    output_dtype: torch.dtype,
) -> str:
    output_name = "null_conditioning.safetensors"
    output_path = output_dir / output_name
    if args.resume and output_path.exists():
        try:
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                if {"hidden_states", "input_ids", "attention_mask"}.issubset(handle.keys()):
                    return output_name
        except Exception:
            pass
        output_path.unlink(missing_ok=True)
    elif output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Use --resume or a new output directory.")

    device = torch.device(args.device)
    tokens = tokenizer(
        "",
        max_length=settings["model_max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden_states = text_encoder(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
    payload = {
        "hidden_states": hidden_states.to(device="cpu", dtype=output_dtype).contiguous(),
        "input_ids": tokens.input_ids.to(device="cpu", dtype=torch.int32).contiguous(),
        "attention_mask": tokens.attention_mask.to(device="cpu", dtype=torch.uint8).contiguous(),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    save_file(payload, str(temporary_path), metadata={"kind": "sana_null_conditioning"})
    os.replace(temporary_path, output_path)
    print(f"Wrote {output_name}")
    return output_name


def export_split(
    name: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    args: argparse.Namespace,
    output_dtype: torch.dtype,
) -> list[str]:
    if not records:
        return []
    shard_paths = []
    device = torch.device(args.device)
    for shard_start in range(0, len(records), args.shard_size):
        shard_end = min(shard_start + args.shard_size, len(records))
        shard_name = f"{name}-{shard_start:06d}-{shard_end:06d}.safetensors"
        shard_path = output_dir / shard_name
        shard_paths.append(shard_name)
        expected_rows = shard_end - shard_start
        if args.resume and valid_completed_shard(shard_path, expected_rows, args.representation):
            print(f"[{name}] keeping completed shard {shard_name}")
            continue
        if shard_path.exists():
            if args.resume:
                print(f"[{name}] replacing incomplete shard {shard_name}")
                shard_path.unlink()
            else:
                raise FileExistsError(
                    f"Output shard already exists: {shard_path}. Use --resume or a new output directory."
                )

        payload_parts: dict[str, list[torch.Tensor]] = {}
        for batch_start in range(shard_start, shard_end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, shard_end)
            prompts = [record["prompt"] for record in records[batch_start:batch_end]]
            with torch.inference_mode():
                hidden_states, input_ids, attention_mask = selected_conditioning(
                    prompts, tokenizer, text_encoder, settings, device
                )
                payload = build_payload(
                    hidden_states,
                    input_ids,
                    attention_mask,
                    torch.arange(batch_start, batch_end),
                    args.representation,
                    output_dtype,
                )
            for key, tensor in payload.items():
                payload_parts.setdefault(key, []).append(tensor)

        combined = {key: torch.cat(parts, dim=0).contiguous() for key, parts in payload_parts.items()}
        temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        temporary_path.unlink(missing_ok=True)
        save_file(
            combined,
            str(temporary_path),
            metadata={
                "split": name,
                "representation": args.representation,
                "model_id": settings["model_id"],
                "model_max_length": str(settings["model_max_length"]),
            },
        )
        os.replace(temporary_path, shard_path)
        print(f"[{name}] wrote {shard_name} ({shard_end:,}/{len(records):,})")
    return shard_paths


def main() -> None:
    args = parse_args()
    settings = load_text_settings(args.config, args.model_id)
    train_records = read_jsonl(args.train_metadata)
    eval_records = read_jsonl(args.eval_metadata)
    splits, stats = build_splits(
        train_records,
        eval_records,
        fit_count=args.fit_prompts,
        validation_count=args.validation_prompts,
        seed=args.seed,
    )
    if not args.include_official_eval:
        splits["official_eval"] = []
    split_summary(splits, stats)
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --resume or a new directory.")

    run_config = {
        "format_version": 1,
        "config": str(args.config),
        "train_metadata": str(args.train_metadata),
        "eval_metadata": str(args.eval_metadata),
        "fit_prompts": len(splits["fit"]),
        "validation_prompts": len(splits["validation"]),
        "official_eval_prompts": len(splits["official_eval"]),
        "seed": args.seed,
        "shard_size": args.shard_size,
        "representation": args.representation,
        "output_dtype": args.output_dtype,
        "encoder_name": settings["encoder_name"],
        "model_id": settings["model_id"],
        "model_max_length": settings["model_max_length"],
        "chi_prompt": settings["chi_prompt"],
        "config_sha256": file_sha256(args.config),
        "train_metadata_sha256": file_sha256(args.train_metadata),
        "eval_metadata_sha256": file_sha256(args.eval_metadata),
        "source_stats": stats,
    }
    run_config_path = output_dir / "run_config.json"
    if args.resume and run_config_path.exists():
        with run_config_path.open(encoding="utf-8") as handle:
            previous_config = json.load(handle)
        if previous_config != run_config:
            raise ValueError("Existing run_config.json does not match this invocation; use a new output directory")
    else:
        temporary_path = run_config_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(run_config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, run_config_path)

    for name, records in splits.items():
        manifest_path = output_dir / f"{name}_prompts.jsonl"
        if not (args.resume and manifest_path.exists()):
            write_prompt_manifest(manifest_path, records)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    print(f"Loading {settings['model_id']} in bfloat16 on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"])
    tokenizer.padding_side = "right"
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(settings["model_id"], torch_dtype=torch.bfloat16)
        .get_decoder()
        .to(args.device)
        .eval()
    )
    text_encoder.requires_grad_(False)

    output_dtype = OUTPUT_DTYPES[args.output_dtype]
    null_conditioning = export_null_conditioning(
        output_dir,
        tokenizer,
        text_encoder,
        settings,
        args,
        output_dtype,
    )
    shard_index = {}
    for name, records in splits.items():
        shard_index[name] = export_split(
            name,
            records,
            output_dir,
            tokenizer,
            text_encoder,
            settings,
            args,
            output_dtype,
        )

    manifest = dict(run_config)
    manifest["hidden_size"] = int(text_encoder.config.hidden_size)
    manifest["model_commit"] = getattr(text_encoder.config, "_commit_hash", None)
    manifest["null_conditioning"] = null_conditioning
    manifest["shards"] = shard_index
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)
    print(f"Export complete: {manifest_path}")


if __name__ == "__main__":
    main()
