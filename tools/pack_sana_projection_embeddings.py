#!/usr/bin/env python3
"""Pack sharded SANA token states into one calculatingW-compatible .pt file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/text_embeddings/sana"),
        help="Directory created by extract_sana_text_embeddings.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/text_embeddings/sana_projection_embeddings.pt"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_prompt_mask(
    prompts: list[str],
    tokenizer,
    chi_prompt: str,
    model_max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conditioned = [chi_prompt + prompt for prompt in prompts] if chi_prompt else prompts
    if chi_prompt:
        tokenizer_max_length = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
    else:
        tokenizer_max_length = model_max_length

    tokens = tokenizer(
        conditioned,
        max_length=tokenizer_max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    select_index = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.arange(tokenizer_max_length - model_max_length + 1, tokenizer_max_length),
        )
    )
    input_ids = tokens.input_ids.index_select(1, select_index)
    attention_mask = tokens.attention_mask.index_select(1, select_index).bool()
    offsets = tokens.offset_mapping.index_select(1, select_index)

    prefix_length = len(chi_prompt)
    prompt_mask = torch.zeros_like(attention_mask)
    for row, prompt in enumerate(prompts):
        prompt_end = prefix_length + len(prompt)
        starts = offsets[row, :, 0]
        ends = offsets[row, :, 1]
        overlaps_prompt = (ends > prefix_length) & (starts < prompt_end) & (ends > starts)
        prompt_mask[row] = attention_mask[row] & overlaps_prompt
        if not bool(prompt_mask[row].any()):
            raise ValueError(f"No prompt tokens remained after SANA selection for prompt {prompt!r}")
    return prompt_mask, input_ids, attention_mask


def pool_split(
    split: str,
    input_dir: Path,
    prompt_rows: list[dict[str, Any]],
    shard_names: list[str],
    tokenizer,
    chi_prompt: str,
    model_max_length: int,
) -> torch.Tensor:
    pooled_parts = []
    expected_next_row = 0
    for shard_name in shard_names:
        shard_path = input_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing {split} shard: {shard_path}")
        shard = load_file(str(shard_path), device="cpu")
        if "hidden_states" not in shard:
            raise KeyError(
                f"{shard_path} has no hidden_states. The source export must use --representation tokens or both."
            )

        row_indices = shard["row_indices"].long()
        expected_indices = torch.arange(expected_next_row, expected_next_row + len(row_indices))
        if not torch.equal(row_indices, expected_indices):
            raise ValueError(f"Unexpected row ordering in {shard_path}")
        batch_rows = prompt_rows[expected_next_row : expected_next_row + len(row_indices)]
        prompts = [row["prompt"] for row in batch_rows]
        prompt_mask, expected_ids, expected_attention = selected_prompt_mask(
            prompts,
            tokenizer,
            chi_prompt,
            model_max_length,
        )
        if not torch.equal(shard["input_ids"].long(), expected_ids):
            raise ValueError(f"Tokenizer IDs do not match the saved embeddings in {shard_path}")
        if not torch.equal(shard["attention_mask"].bool(), expected_attention):
            raise ValueError(f"Attention masks do not match the saved embeddings in {shard_path}")

        hidden = shard["hidden_states"].float()
        weights = prompt_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        pooled_parts.append(pooled)
        expected_next_row += len(row_indices)
        print(f"[{split}] pooled {expected_next_row:,}/{len(prompt_rows):,}")

    if expected_next_row != len(prompt_rows):
        raise ValueError(f"{split} has {expected_next_row} embedding rows but {len(prompt_rows)} prompt rows")
    return torch.cat(pooled_parts, dim=0)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    manifest = read_json(input_dir / "manifest.json")
    chi_prompt = manifest.get("chi_prompt", "")
    model_max_length = int(manifest["model_max_length"])
    model_id = manifest["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    split_payloads = {}
    for split in ("fit", "validation", "official_eval"):
        prompt_rows = read_jsonl(input_dir / f"{split}_prompts.jsonl")
        embeddings = pool_split(
            split,
            input_dir,
            prompt_rows,
            manifest["shards"][split],
            tokenizer,
            chi_prompt,
            model_max_length,
        )
        split_payloads[split] = {
            "rows": prompt_rows,
            "ids": [row["prompt_sha256"] for row in prompt_rows],
            "texts": [row["prompt"] for row in prompt_rows],
            "embeddings": embeddings,
        }

    fit = split_payloads["fit"]
    validation = split_payloads["validation"]
    official_eval = split_payloads["official_eval"]
    bundle = {
        "kind": "sana_text_projection",
        "model_id": model_id,
        "representation": "last_hidden_state_prompt_token_masked_mean",
        "hidden_dim": int(fit["embeddings"].shape[1]),
        # These two keys match the older calculatingW embedding bundles.
        "class_names": fit["texts"],
        "embeddings": fit["embeddings"],
        "ids": fit["ids"],
        "rows": fit["rows"],
        # Held-out calculation validation.
        "validation_class_names": validation["texts"],
        "validation_embeddings": validation["embeddings"],
        "validation_ids": validation["ids"],
        "validation_rows": validation["rows"],
        # Never use these targets while fitting or choosing projection settings.
        "official_eval_class_names": official_eval["texts"],
        "official_eval_embeddings": official_eval["embeddings"],
        "official_eval_ids": official_eval["ids"],
        "official_eval_rows": official_eval["rows"],
        "source_manifest": manifest,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    torch.save(bundle, temporary_path)
    os.replace(temporary_path, output_path)

    print(f"Saved projection bundle: {output_path}")
    print(f"fit embeddings: {tuple(bundle['embeddings'].shape)}")
    print(f"validation embeddings: {tuple(bundle['validation_embeddings'].shape)}")
    print(f"official eval embeddings: {tuple(bundle['official_eval_embeddings'].shape)}")


if __name__ == "__main__":
    main()
