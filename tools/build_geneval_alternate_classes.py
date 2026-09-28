#!/usr/bin/env python3
"""Clone GenEval prompt structures while replacing only object classes."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_METADATA = Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl")
DEFAULT_ORIGINAL_CLASSES = Path("tools/metrics/geneval/evaluation/object_names.txt")
DEFAULT_ALTERNATE_CLASSES = Path("tools/metrics/geneval/prompts/alternate_object_names.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--original-classes", type=Path, default=DEFAULT_ORIGINAL_CLASSES)
    parser.add_argument("--alternate-classes", type=Path, default=DEFAULT_ALTERNATE_CLASSES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/text_embeddings/geneval_alternate_classes"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"Missing prompt at {path}:{line_number}")
            rows.append(row)
    return rows


def read_original_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_alternate_classes(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not all(field.strip() for field in fields):
                raise ValueError(f"Expected singular<TAB>plural at {path}:{line_number}")
            rows.append((fields[0].strip(), fields[1].strip()))
    return rows


def article(phrase: str) -> str:
    return "an" if phrase[0].lower() in "aeiou" else "a"


def noun_phrase(class_name: str, color: str | None = None) -> str:
    phrase = f"{color} {class_name}" if color else class_name
    return f"{article(phrase)} {phrase}"


def number_word(value: int) -> str:
    words = {2: "two", 3: "three", 4: "four"}
    if value not in words:
        raise ValueError(f"Unsupported GenEval count: {value}")
    return words[value]


def rebuild_prompt(row: dict[str, Any], plural_by_class: dict[str, str]) -> str:
    tag = row["tag"]
    include = row["include"]
    if tag == "single_object":
        return f"a photo of {noun_phrase(include[0]['class'])}"
    if tag == "two_object":
        return f"a photo of {noun_phrase(include[0]['class'])} and {noun_phrase(include[1]['class'])}"
    if tag == "counting":
        item = include[0]
        return f"a photo of {number_word(int(item['count']))} {plural_by_class[item['class']]}"
    if tag == "colors":
        item = include[0]
        return f"a photo of {noun_phrase(item['class'], item['color'])}"
    if tag == "color_attr":
        first, second = include
        return (
            f"a photo of {noun_phrase(first['class'], first['color'])} and "
            f"{noun_phrase(second['class'], second['color'])}"
        )
    if tag == "position":
        positioned = [(index, item) for index, item in enumerate(include) if item.get("position")]
        if len(positioned) != 1:
            raise ValueError(f"Expected one positioned object: {row}")
        _, item = positioned[0]
        relation, reference_index = item["position"]
        reference = include[int(reference_index)]
        return f"a photo of {noun_phrase(item['class'])} {relation} {noun_phrase(reference['class'])}"
    raise ValueError(f"Unsupported GenEval tag: {tag!r}")


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    original_classes = read_original_classes(args.original_classes)
    alternate_classes = read_alternate_classes(args.alternate_classes)
    if len(original_classes) != len(alternate_classes):
        raise ValueError(
            f"Class counts differ: {len(original_classes)} original vs "
            f"{len(alternate_classes)} alternate"
        )
    if len(set(original_classes)) != len(original_classes):
        raise ValueError("Original class list contains duplicates")
    alternate_singulars = [singular for singular, _ in alternate_classes]
    if len(set(alternate_singulars)) != len(alternate_singulars):
        raise ValueError("Alternate class list contains duplicates")
    overlap = sorted(set(original_classes).intersection(alternate_singulars))
    if overlap:
        raise ValueError(f"Alternate classes overlap GenEval classes: {overlap}")

    class_map = dict(zip(original_classes, alternate_singulars, strict=True))
    plural_by_class = dict(alternate_classes)
    source_rows = read_jsonl(args.metadata)
    source_used = {item["class"] for row in source_rows for item in row.get("include", [])}
    unknown = sorted(source_used.difference(class_map))
    if unknown:
        raise ValueError(f"Metadata contains classes absent from the original class list: {unknown}")

    output_rows = []
    for source_row in source_rows:
        row = deepcopy(source_row)
        for field in ("include", "exclude"):
            for item in row.get(field, []):
                item["class"] = class_map[item["class"]]
        row["prompt"] = rebuild_prompt(row, plural_by_class)
        output_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "evaluation_metadata_alternate_classes.jsonl"
    mapping_path = output_dir / "class_mapping.json"
    existing = [path for path in (metadata_path, mapping_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Output already exists: {existing}; pass --force to replace it")

    metadata_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    mapping = {
        "format_version": 1,
        "kind": "geneval_disjoint_object_class_mapping",
        "source_metadata": str(args.metadata),
        "prompt_count": len(output_rows),
        "class_count": len(class_map),
        "mapping": [
            {"original": original, "alternate": class_map[original], "alternate_plural": plural}
            for original, (_, plural) in zip(original_classes, alternate_classes, strict=True)
        ],
    }
    atomic_write(metadata_path, metadata_text)
    atomic_write(mapping_path, json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")

    print(f"Prompts: {len(output_rows)}")
    print(f"Disjoint object classes: {len(class_map)}")
    print(f"Metadata: {metadata_path}")
    print(f"Mapping: {mapping_path}")
    print(f"First prompt: {output_rows[0]['prompt']}")


if __name__ == "__main__":
    main()
