#!/usr/bin/env python3
"""Add disjoint ImageNet-21K classes to the exact GenEval calculation set.

The existing alternate-class GenEval rows are retained unchanged. Each new
class contributes one prompt using GenEval's native single-object structure,
``a photo of a/an <class>``. Official GenEval classes are excluded so the
benchmark remains held out. The pinned ImageNet-21K label list is downloaded
with the Python standard library and verified by SHA-256; no package install or
model weights are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OFFICIAL_METADATA = Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl")
DEFAULT_BASE_METADATA = Path(
    "output/text_embeddings/geneval_alternate_classes/evaluation_metadata_alternate_classes.jsonl"
)
DEFAULT_ALTERNATE_CLASSES = Path("tools/metrics/geneval/prompts/alternate_object_names.tsv")
IMAGENET21K_LABELS_URL = (
    "https://gist.githubusercontent.com/BIGBALLON/63d153b966f64a7dbf7a06d7a28396c2/raw/"
    "imagenet21k_ids_with_classnames.csv"
)
IMAGENET21K_LABELS_SHA256 = "66e637be9c3dc9c6a3850ec04ceae807d7225aece331581a72b5ca681016829d"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-metadata", type=Path, default=DEFAULT_BASE_METADATA)
    parser.add_argument("--official-metadata", type=Path, default=DEFAULT_OFFICIAL_METADATA)
    parser.add_argument("--alternate-classes", type=Path, default=DEFAULT_ALTERNATE_CLASSES)
    parser.add_argument("--additional-classes", type=int, default=1_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/text_embeddings/geneval_augmented_1000"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.additional_classes <= 0:
        parser.error("--additional-classes must be positive")
    return args


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip().lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            rows.append(row)
    return rows


def read_alternate_classes(path: Path) -> set[str]:
    classes = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Expected singular<TAB>plural at {path}:{line_number}")
            classes.add(normalize(fields[0]))
    return classes


def imagenet21k_classes() -> list[tuple[str, set[str]]]:
    with urllib.request.urlopen(IMAGENET21K_LABELS_URL, timeout=60) as response:
        payload = response.read()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != IMAGENET21K_LABELS_SHA256:
        raise RuntimeError(
            "ImageNet-21K label checksum changed: "
            f"expected {IMAGENET21K_LABELS_SHA256}, got {actual_sha256}"
        )

    candidates = []
    text = payload.decode("utf-8")
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or not row[0].startswith("n"):
            continue
        aliases = {normalize(alias) for alias in row[1:] if normalize(alias)}
        if aliases:
            candidates.append((normalize(row[1]), aliases))
    # The source follows WordNet taxonomy order. A fixed shuffle prevents the
    # first N labels from being dominated by one branch while remaining fully
    # reproducible.
    random.Random(0).shuffle(candidates)
    return candidates


def official_blocklist(rows: list[dict[str, Any]]) -> set[str]:
    return {
        normalize(item["class"])
        for row in rows
        for field in ("include", "exclude")
        for item in row.get(field, [])
    }


def valid_visual_label(label: str) -> bool:
    return (
        2 <= len(label) <= 40
        and len(label.split()) <= 3
        and re.fullmatch(r"[a-z][a-z -]*", label) is not None
    )


def select_classes(
    candidates: list[tuple[str, set[str]]], blocked: set[str], count: int
) -> list[str]:
    selected = []
    seen = set(blocked)
    for label, aliases in candidates:
        if aliases.intersection(blocked) or label in seen or not valid_visual_label(label):
            continue
        selected.append(label)
        seen.update(aliases)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected):,} eligible disjoint ImageNet-21K classes were available; "
        f"requested {count:,}"
    )


def article(label: str) -> str:
    return "an" if label[0] in "aeiou" else "a"


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base_metadata)
    official_rows = read_jsonl(args.official_metadata)

    blocked = official_blocklist(official_rows)
    blocked.update(read_alternate_classes(args.alternate_classes))
    blocked.update(
        normalize(item["class"])
        for row in base_rows
        for item in row.get("include", [])
    )
    additional_classes = select_classes(imagenet21k_classes(), blocked, args.additional_classes)

    added_rows = [
        {
            "tag": "calculation_extra_single_object",
            "include": [{"class": class_name, "count": 1}],
            "prompt": f"a photo of {article(class_name)} {class_name}",
        }
        for class_name in additional_classes
    ]
    combined_rows = base_rows + added_rows

    official_classes = {
        normalize(item["class"])
        for row in official_rows
        for item in row.get("include", [])
    }
    combined_classes = {
        normalize(item["class"])
        for row in combined_rows
        for item in row.get("include", [])
    }
    leaked = sorted(official_classes.intersection(combined_classes))
    if leaked:
        raise RuntimeError(f"Official classes leaked into calculation data: {leaked}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "evaluation_metadata_augmented_classes.jsonl"
    manifest_path = output_dir / "augmentation_manifest.json"
    existing = [path for path in (metadata_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Output already exists: {existing}; pass --force to replace it")

    atomic_write(
        metadata_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows),
    )
    manifest = {
        "format_version": 1,
        "kind": "geneval_augmented_class_calculation_set",
        "class_source": IMAGENET21K_LABELS_URL,
        "class_source_sha256": IMAGENET21K_LABELS_SHA256,
        "class_selection": "deterministic_shuffle_seed_0_then_filter",
        "base_metadata": str(args.base_metadata),
        "official_metadata": str(args.official_metadata),
        "base_prompt_count": len(base_rows),
        "additional_prompt_count": len(added_rows),
        "total_prompt_count": len(combined_rows),
        "base_unique_class_count": len(combined_classes.difference(additional_classes)),
        "additional_class_count": len(additional_classes),
        "total_unique_class_count": len(combined_classes),
        "additional_classes": additional_classes,
        "prompt_template": "a photo of a/an <class>",
        "official_class_overlap": leaked,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Base prompts: {len(base_rows):,}")
    print(f"Additional visual classes/prompts: {len(added_rows):,}")
    print(f"Total fitting prompts: {len(combined_rows):,}")
    print(f"Total unique calculation classes: {len(combined_classes):,}")
    print(f"Official class overlap: {len(leaked)}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"First additional classes: {additional_classes[:10]}")


if __name__ == "__main__":
    main()
