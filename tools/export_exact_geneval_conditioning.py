#!/usr/bin/env python3
"""Export the exact native SANA conditioning used for official GenEval prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CONFIG = Path("configs/sana_config/1024ms/Sana_600M_img1024.yaml")
DEFAULT_METADATA = Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl")
TEXT_ENCODERS = {
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=Path("output/text_embeddings/geneval_exact"))
    parser.add_argument("--model-id", help="Override the text encoder selected by the config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_kaggle_inputs(path: Path, rows: list[dict], chi_prompt: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            prompt = row["prompt"]
            payload = {
                "index": index,
                "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "conditioned_text": chi_prompt + prompt,
                "tag": row.get("tag"),
                "include": row.get("include", []),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    text_config = config["text_encoder"]
    encoder_name = text_config["text_encoder_name"]
    model_id = args.model_id or TEXT_ENCODERS.get(encoder_name)
    if not model_id:
        raise ValueError(f"Unknown text encoder {encoder_name!r}; pass --model-id")

    sequence_length = int(text_config["model_max_length"])
    chi_prompt = "\n".join(text_config.get("chi_prompt") or [])
    rows = read_jsonl(args.metadata)
    prompts = [row["prompt"] for row in rows]
    conditioned_texts = [chi_prompt + prompt for prompt in prompts]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning_path = output_dir / "geneval_native_exact.pt"
    kaggle_path = output_dir / "geneval_exact_inputs.jsonl"
    if not args.force:
        existing = [str(path) for path in (conditioning_path, kaggle_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing}; pass --force to replace it")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer_max_length = len(tokenizer.encode(chi_prompt)) + sequence_length - 2 if chi_prompt else sequence_length
    encoded = tokenizer(
        conditioned_texts,
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

    selected_input_ids = encoded["input_ids"].index_select(1, select_index).to(torch.int32)
    selected_attention_mask = encoded["attention_mask"].index_select(1, select_index).to(torch.uint8)
    selected_special_mask = encoded["special_tokens_mask"].index_select(1, select_index).to(torch.uint8)
    selected_offsets = encoded["offset_mapping"].index_select(1, select_index).to(torch.int32)

    device = torch.device(args.device)
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        .get_decoder()
        .to(device)
        .eval()
    )
    text_encoder.requires_grad_(False)
    selected_device_index = select_index.to(device)
    embedding_parts = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="Embedding exact GenEval inputs"):
            end = min(start + args.batch_size, len(rows))
            hidden = text_encoder(
                encoded["input_ids"][start:end].to(device),
                encoded["attention_mask"][start:end].to(device),
            )[0]
            embedding_parts.append(
                hidden.index_select(1, selected_device_index).to(device="cpu", dtype=torch.bfloat16)
            )
    native_embeddings = torch.cat(embedding_parts).contiguous()

    prompt_ids = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    bundle = {
        "format_version": 1,
        "kind": "sana_native_exact_geneval_conditioning",
        "projected_text_mode": "native_sequence",
        # Y_pred makes this file directly usable by --projected_text_embeddings.
        "Y_pred": native_embeddings,
        "attention_mask": selected_attention_mask.contiguous(),
        "input_ids": selected_input_ids.contiguous(),
        "special_tokens_mask": selected_special_mask.contiguous(),
        "offset_mapping": selected_offsets.contiguous(),
        "prompt_texts": prompts,
        "conditioned_texts": conditioned_texts,
        "ids": prompt_ids,
        "chi_prompt": chi_prompt,
        "text_encoder_name": encoder_name,
        "text_encoder_model_id": model_id,
        "tokenizer_padding_side": "right",
        "sequence_length": sequence_length,
        "tokenizer_max_length": tokenizer_max_length,
        "config_path": str(args.config),
        "metadata_path": str(args.metadata),
    }
    atomic_torch_save(conditioning_path, bundle)
    write_kaggle_inputs(kaggle_path, rows, chi_prompt)

    saved = torch.load(conditioning_path, map_location="cpu", weights_only=True, mmap=True)
    expected_shape = (len(rows), sequence_length, native_embeddings.shape[-1])
    if tuple(saved["Y_pred"].shape) != expected_shape:
        raise RuntimeError(f"Saved embedding shape is wrong: {tuple(saved['Y_pred'].shape)} != {expected_shape}")
    if not torch.equal(saved["attention_mask"], selected_attention_mask):
        raise RuntimeError("Saved attention mask differs from the native SANA mask")

    print(f"Conditioning: {conditioning_path}")
    print(f"Kaggle input: {kaggle_path}")
    print(f"Y_pred: {tuple(native_embeddings.shape)} {native_embeddings.dtype}")
    print(f"Attention mask: {tuple(selected_attention_mask.shape)} ({int(selected_attention_mask.sum())} active)")
    print(f"CHI characters: {len(chi_prompt)}")


if __name__ == "__main__":
    main()
