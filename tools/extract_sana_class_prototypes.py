#!/usr/bin/env python3
"""Extract SANA class prototypes and untouched native GenEval conditioning.

Calculation classes are encoded through several templates.  Only the hidden
states overlapping the inserted class name are pooled; normalized template
vectors are averaged into one prototype per semantic class.  Official GenEval
prompts are exported separately as the exact 300-position SANA conditioning
sequence and are never included in the class prototypes.
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
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
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
    parser.add_argument("--class-spec", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sana_config/1024ms/Sana_600M_img1024.yaml"),
    )
    parser.add_argument(
        "--eval-metadata",
        type=Path,
        default=Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--class-shard-size", type=int, default=16)
    parser.add_argument("--eval-shard-size", type=int, default=32)
    parser.add_argument("--output-dtype", choices=tuple(OUTPUT_DTYPES), default="float16")
    parser.add_argument(
        "--include-official-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "class_shard_size", "eval_shard_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
            rows.append({**row, "prompt": prompt.strip()})
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_text_settings(config_path: Path, model_override: str | None) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    text_config = config["text_encoder"]
    encoder_name = text_config["text_encoder_name"]
    model_id = model_override or TEXT_ENCODER_IDS.get(encoder_name)
    if not model_id:
        raise ValueError(f"Cannot resolve SANA text encoder {encoder_name!r}; pass --model-id")
    chi_lines = text_config.get("chi_prompt") or []
    if not isinstance(chi_lines, list) or not all(isinstance(line, str) for line in chi_lines):
        raise ValueError("text_encoder.chi_prompt must be a list of strings")
    return {
        "encoder_name": encoder_name,
        "model_id": model_id,
        "model_max_length": int(text_config.get("model_max_length", 300)),
        "chi_prompt": "\n".join(chi_lines),
    }


def validate_class_spec(spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    if spec.get("kind") != "sana_projection_class_set":
        raise ValueError("--class-spec is not a sana_projection_class_set")
    class_names = spec.get("class_names")
    templates = spec.get("templates")
    if not isinstance(class_names, list) or not class_names or not all(isinstance(x, str) for x in class_names):
        raise ValueError("class-spec class_names must be a non-empty string list")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class-spec contains duplicate class names")
    if not isinstance(templates, list) or not templates or not all(isinstance(x, str) for x in templates):
        raise ValueError("class-spec templates must be a non-empty string list")
    if any(template.count("{}") != 1 for template in templates):
        raise ValueError("Every template must contain exactly one '{}' placeholder")
    return class_names, templates


def format_prompt(template: str, class_name: str) -> tuple[str, tuple[int, int]]:
    prefix, _ = template.split("{}")
    prompt = template.format(class_name)
    start = len(prefix)
    return prompt, (start, start + len(class_name))


@torch.inference_mode()
def selected_conditioning(
    prompts: list[str],
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    device: torch.device,
    fixed_length: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chi_prompt = settings["chi_prompt"]
    conditioned = [chi_prompt + prompt for prompt in prompts] if chi_prompt else prompts
    model_max_length = settings["model_max_length"]
    if chi_prompt:
        tokenizer_max_length = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
    else:
        tokenizer_max_length = model_max_length

    tokens = tokenizer(
        conditioned,
        max_length=tokenizer_max_length,
        padding="max_length" if fixed_length else "longest",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = tokens.pop("offset_mapping")
    tokens = tokens.to(device)
    hidden = text_encoder(
        tokens.input_ids,
        attention_mask=tokens.attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    if fixed_length:
        select_device = torch.cat(
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
        select_cpu = select_device.cpu()
        hidden = hidden.index_select(1, select_device)
        input_ids = input_ids.index_select(1, select_device)
        attention_mask = attention_mask.index_select(1, select_device)
        offsets = offsets.index_select(1, select_cpu)
    return hidden, input_ids, attention_mask, offsets


def pool_class_spans(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets: torch.Tensor,
    spans: list[tuple[int, int]],
    chi_length: int,
) -> torch.Tensor:
    offsets = offsets.to(hidden.device)
    masks = []
    for row, (start, end) in enumerate(spans):
        start += chi_length
        end += chi_length
        token_start = offsets[row, :, 0]
        token_end = offsets[row, :, 1]
        mask = (
            attention_mask[row].bool()
            & (token_end > start)
            & (token_start < end)
            & (token_end > token_start)
        )
        if not bool(mask.any()):
            raise ValueError(f"No selected token overlaps class character span {spans[row]}")
        masks.append(mask)
    class_mask = torch.stack(masks)
    # Pool in float32 so the class prototypes do not accumulate bfloat16
    # rounding error; only the saved shards are cast to --output-dtype.
    weights = class_mask.unsqueeze(-1).float()
    hidden_float = hidden.float()
    return (hidden_float * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def valid_shard(path: Path, required: set[str], expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return required.issubset(handle.keys()) and handle.get_tensor("row_indices").shape[0] == expected_rows
    except Exception:
        return False


def save_tensor_shard(path: Path, payload: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    save_file({key: value.contiguous() for key, value in payload.items()}, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def export_class_prototypes(
    class_names: list[str],
    templates: list[str],
    output_dir: Path,
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    device = torch.device(args.device)
    output_dtype = OUTPUT_DTYPES[args.output_dtype]
    shard_names = []
    required = {"row_indices", "embeddings", "unit_embeddings", "mean_norms"}

    for shard_start in range(0, len(class_names), args.class_shard_size):
        shard_end = min(shard_start + args.class_shard_size, len(class_names))
        shard_name = f"classes-{shard_start:06d}-{shard_end:06d}.safetensors"
        shard_path = output_dir / shard_name
        shard_names.append(shard_name)
        expected_rows = shard_end - shard_start
        if args.resume and valid_shard(shard_path, required, expected_rows):
            print(f"[classes] keeping {shard_name}")
            continue
        if shard_path.exists() and not args.resume:
            raise FileExistsError(f"Output already exists: {shard_path}; pass --resume")

        prompts = []
        spans = []
        local_class_ids = []
        for local_id, class_name in enumerate(class_names[shard_start:shard_end]):
            for template in templates:
                prompt, span = format_prompt(template, class_name)
                prompts.append(prompt)
                spans.append(span)
                local_class_ids.append(local_id)

        prototype_sum = None
        norm_sum = torch.zeros(expected_rows, dtype=torch.float32)
        counts = torch.zeros(expected_rows, dtype=torch.float32)
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            hidden, _, attention_mask, offsets = selected_conditioning(
                prompts[batch_start:batch_end],
                tokenizer,
                text_encoder,
                settings,
                device,
                fixed_length=False,
            )
            vectors = pool_class_spans(
                hidden,
                attention_mask,
                offsets,
                spans[batch_start:batch_end],
                len(settings["chi_prompt"]),
            ).float()
            norms = vectors.norm(dim=1).clamp_min(1e-12)
            unit = (vectors / norms[:, None]).cpu()
            ids = torch.tensor(local_class_ids[batch_start:batch_end], dtype=torch.long)
            if prototype_sum is None:
                prototype_sum = torch.zeros(expected_rows, vectors.shape[-1], dtype=torch.float32)
            prototype_sum.index_add_(0, ids, unit)
            norm_sum.index_add_(0, ids, norms.cpu())
            counts.index_add_(0, ids, torch.ones_like(norms.cpu()))

        if prototype_sum is None or not bool((counts == len(templates)).all()):
            raise RuntimeError(f"Incomplete template aggregation for class shard {shard_start}:{shard_end}")
        unit_prototypes = F.normalize(prototype_sum / counts[:, None], dim=1)
        mean_norms = norm_sum / counts
        prototypes = unit_prototypes * mean_norms[:, None]
        save_tensor_shard(
            shard_path,
            {
                "row_indices": torch.arange(shard_start, shard_end, dtype=torch.int32),
                "embeddings": prototypes.to(output_dtype),
                "unit_embeddings": unit_prototypes.to(output_dtype),
                "mean_norms": mean_norms,
            },
            {
                "kind": "sana_class_prototypes",
                "model_id": settings["model_id"],
                "pooling": "class_token_span_then_normalized_template_mean",
            },
        )
        print(f"[classes] wrote {shard_name} ({shard_end:,}/{len(class_names):,})")
    return shard_names


def pack_class_prototypes(
    class_names: list[str], templates: list[str], shard_names: list[str], output_dir: Path, settings: dict[str, Any]
) -> str:
    parts: dict[str, list[torch.Tensor]] = {"embeddings": [], "unit_embeddings": [], "mean_norms": []}
    expected_index = 0
    for shard_name in shard_names:
        shard = load_file(str(output_dir / shard_name), device="cpu")
        indices = shard["row_indices"].long()
        expected = torch.arange(expected_index, expected_index + len(indices))
        if not torch.equal(indices, expected):
            raise ValueError(f"Unexpected class ordering in {shard_name}")
        for key in parts:
            parts[key].append(shard[key])
        expected_index += len(indices)
    if expected_index != len(class_names):
        raise ValueError(f"Packed {expected_index} class rows; expected {len(class_names)}")

    bundle = {
        "kind": "sana_class_prototypes",
        "model_id": settings["model_id"],
        "representation": "last_hidden_state_class_token_span_normalized_template_mean",
        "class_names": class_names,
        "templates": templates,
        "embeddings": torch.cat(parts["embeddings"]).float(),
        "unit_embeddings": torch.cat(parts["unit_embeddings"]).float(),
        "mean_norms": torch.cat(parts["mean_norms"]).float(),
        "model_max_length": settings["model_max_length"],
    }
    output_name = "sana_class_prototypes.pt"
    output_path = output_dir / output_name
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, output_path)
    print(f"Packed class prototype file: {output_path}")
    print(f"Class embedding shape: {tuple(bundle['embeddings'].shape)}")
    return output_name


def write_eval_prompt_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> str:
    output_name = "official_eval_prompts.jsonl"
    output_path = output_dir / output_name
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            prompt = row["prompt"]
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "tag": row.get("tag"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temporary, output_path)
    return output_name


def export_official_eval(
    rows: list[dict[str, Any]],
    output_dir: Path,
    tokenizer,
    text_encoder,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    device = torch.device(args.device)
    output_dtype = OUTPUT_DTYPES[args.output_dtype]
    shard_names = []
    required = {"row_indices", "hidden_states", "input_ids", "attention_mask", "offset_mapping"}
    for shard_start in range(0, len(rows), args.eval_shard_size):
        shard_end = min(shard_start + args.eval_shard_size, len(rows))
        shard_name = f"official-eval-{shard_start:06d}-{shard_end:06d}.safetensors"
        shard_path = output_dir / shard_name
        shard_names.append(shard_name)
        expected_rows = shard_end - shard_start
        if args.resume and valid_shard(shard_path, required, expected_rows):
            print(f"[official_eval] keeping {shard_name}")
            continue
        if shard_path.exists() and not args.resume:
            raise FileExistsError(f"Output already exists: {shard_path}; pass --resume")

        payload_parts: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "input_ids": [],
            "attention_mask": [],
            "offset_mapping": [],
        }
        for batch_start in range(shard_start, shard_end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, shard_end)
            prompts = [row["prompt"] for row in rows[batch_start:batch_end]]
            hidden, input_ids, attention_mask, offsets = selected_conditioning(
                prompts, tokenizer, text_encoder, settings, device
            )
            payload_parts["hidden_states"].append(hidden.to(device="cpu", dtype=output_dtype))
            payload_parts["input_ids"].append(input_ids.to(device="cpu", dtype=torch.int32))
            payload_parts["attention_mask"].append(attention_mask.to(device="cpu", dtype=torch.uint8))
            payload_parts["offset_mapping"].append(offsets.to(dtype=torch.int32))
        payload = {key: torch.cat(value) for key, value in payload_parts.items()}
        payload["row_indices"] = torch.arange(shard_start, shard_end, dtype=torch.int32)
        save_tensor_shard(
            shard_path,
            payload,
            {
                "kind": "sana_native_geneval_conditioning",
                "model_id": settings["model_id"],
                "model_max_length": str(settings["model_max_length"]),
            },
        )
        print(f"[official_eval] wrote {shard_name} ({shard_end:,}/{len(rows):,})")
    return shard_names


def main() -> None:
    args = parse_args()
    spec = read_json(args.class_spec)
    class_names, templates = validate_class_spec(spec)
    eval_rows = read_jsonl(args.eval_metadata)
    eval_classes = {item["class"].strip().lower() for row in eval_rows for item in row["include"]}
    leaked = sorted(eval_classes.intersection(name.strip().lower() for name in class_names))
    if leaked:
        raise ValueError(f"Calculation class specification contains GenEval classes: {leaked}")
    settings = load_text_settings(args.config, args.model_id)

    print(f"Calculation classes: {len(class_names):,}")
    print(f"Templates per class: {len(templates)}")
    print(f"Intermediate calculation prompts: {len(class_names) * len(templates):,}")
    print(f"Official GenEval prompts kept separate: {len(eval_rows):,}")
    print(
        f"Native official conditioning shape: [{len(eval_rows)}, {settings['model_max_length']}, hidden_dim]"
    )
    if args.dry_run:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --resume or use a new directory")

    run_config = {
        "format_version": 1,
        "kind": "sana_class_projection_export",
        "class_spec": str(args.class_spec),
        "class_spec_sha256": sha256_file(args.class_spec),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "eval_metadata": str(args.eval_metadata),
        "eval_metadata_sha256": sha256_file(args.eval_metadata),
        "class_count": len(class_names),
        "template_count": len(templates),
        "official_eval_count": len(eval_rows) if args.include_official_eval else 0,
        "model_id": settings["model_id"],
        "model_max_length": settings["model_max_length"],
        "output_dtype": args.output_dtype,
        "class_shard_size": args.class_shard_size,
        "eval_shard_size": args.eval_shard_size,
    }
    run_config_path = output_dir / "manifest.json"
    if args.resume and run_config_path.exists():
        previous_config = read_json(run_config_path)
        mismatched = [key for key, value in run_config.items() if previous_config.get(key) != value]
        if mismatched:
            raise ValueError(f"Existing manifest.json does not match this invocation: {mismatched}")
    else:
        atomic_write_json(run_config_path, run_config)

    print(f"Loading {settings['model_id']} in bfloat16 on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"])
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for class-span and benchmark offset mappings")
    tokenizer.padding_side = "right"
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(settings["model_id"], torch_dtype=torch.bfloat16)
        .get_decoder()
        .to(args.device)
        .eval()
    )
    text_encoder.requires_grad_(False)

    class_shards = export_class_prototypes(
        class_names, templates, output_dir, tokenizer, text_encoder, settings, args
    )
    packed_classes = pack_class_prototypes(class_names, templates, class_shards, output_dir, settings)
    eval_prompt_manifest = None
    eval_shards = []
    if args.include_official_eval:
        eval_prompt_manifest = write_eval_prompt_manifest(output_dir, eval_rows)
        eval_shards = export_official_eval(eval_rows, output_dir, tokenizer, text_encoder, settings, args)

    run_config["class_shards"] = class_shards
    run_config["packed_class_prototypes"] = packed_classes
    run_config["official_eval_prompt_manifest"] = eval_prompt_manifest
    run_config["official_eval_shards"] = eval_shards
    atomic_write_json(run_config_path, run_config)
    print(f"Completed export: {output_dir}")


if __name__ == "__main__":
    main()
