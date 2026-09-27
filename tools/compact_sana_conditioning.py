#!/usr/bin/env python3
"""Losslessly compact a dense SANA conditioning bundle to active tokens."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; pass --force to replace it")

    bundle = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    embeddings = bundle.get("Y_pred")
    attention_mask = bundle.get("attention_mask")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise ValueError("Input bundle must contain dense Y_pred [rows, positions, hidden_dim]")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("Input attention_mask must match the first two Y_pred dimensions")

    active_embeddings = embeddings[attention_mask.bool()].contiguous()
    compact = {key: value for key, value in bundle.items() if key != "Y_pred"}
    compact["Y_pred_active"] = active_embeddings
    compact["conditioning_storage"] = "active_tokens_only"
    compact["dense_shape"] = list(embeddings.shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(compact, temporary)
    os.replace(temporary, args.output)

    saved = torch.load(args.output, map_location="cpu", weights_only=False, mmap=True)
    if not torch.equal(saved["Y_pred_active"], active_embeddings):
        raise RuntimeError("Saved active embeddings differ from the dense source")
    if not torch.equal(saved["attention_mask"], attention_mask):
        raise RuntimeError("Saved attention mask differs from the dense source")

    print(f"Saved: {args.output}")
    print(f"Dense shape: {tuple(embeddings.shape)}")
    print(f"Active shape: {tuple(active_embeddings.shape)}")
    print(f"Size: {args.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
