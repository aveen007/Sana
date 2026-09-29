#!/usr/bin/env python3
"""Repeat exact GenEval prompt structures with disjoint ImageNet-21K classes.

The existing 80-class alternate bank is retained. Additional classes fill the
object slots of repeated original 553-row GenEval banks. Only object class names
(plus their article/plural agreement) are changed; tags, task structure,
attributes, counts, and relations stay native to GenEval. The normal
exact-conditioning exporter supplies the same CHI prefix.
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
DEFAULT_ORIGINAL_CLASSES = Path("tools/metrics/geneval/evaluation/object_names.txt")
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
    parser.add_argument("--original-classes", type=Path, default=DEFAULT_ORIGINAL_CLASSES)
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


def read_original_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [normalize(line) for line in handle if line.strip()]


def read_alternate_classes(path: Path) -> list[tuple[str, str]]:
    classes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Expected singular<TAB>plural at {path}:{line_number}")
            classes.append((normalize(fields[0]), normalize(fields[1])))
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


IRREGULAR_PLURALS = {
    "child": "children",
    "foot": "feet",
    "goose": "geese",
    "man": "men",
    "mouse": "mice",
    "ox": "oxen",
    "person": "people",
    "tooth": "teeth",
    "woman": "women",
}


def pluralize(label: str) -> str:
    words = label.split()
    final = words[-1]
    if final in IRREGULAR_PLURALS:
        words[-1] = IRREGULAR_PLURALS[final]
    elif re.search(r"[^aeiou]y$", final):
        words[-1] = final[:-1] + "ies"
    elif re.search(r"(?:s|x|z|ch|sh)$", final):
        words[-1] = final + "es"
    else:
        words[-1] = final + "s"
    return " ".join(words)


def article(phrase: str) -> str:
    return "an" if phrase[0] in "aeiou" else "a"


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
        positioned = [item for item in include if item.get("position")]
        if len(positioned) != 1:
            raise ValueError(f"Expected one positioned object: {row}")
        item = positioned[0]
        relation, reference_index = item["position"]
        reference = include[int(reference_index)]
        return f"a photo of {noun_phrase(item['class'])} {relation} {noun_phrase(reference['class'])}"
    raise ValueError(f"Unsupported GenEval tag: {tag!r}")


def remap_prompt_bank_by_occurrence(
    source_rows: list[dict[str, Any]],
    replacements: list[str],
    fallback_map: dict[str, str],
    plural_by_class: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from copy import deepcopy

    output_rows = []
    assignments = []
    replacement_index = 0
    for row_index, source_row in enumerate(source_rows):
        row = deepcopy(source_row)
        local_map = {}
        for include_index, item in enumerate(row.get("include", [])):
            original = normalize(item["class"])
            if original not in local_map:
                if replacement_index < len(replacements):
                    replacement = replacements[replacement_index]
                    replacement_index += 1
                    assignments.append(
                        {
                            "row_index": row_index,
                            "include_index": include_index,
                            "original": original,
                            "replacement": replacement,
                        }
                    )
                else:
                    replacement = fallback_map[original]
                local_map[original] = replacement
            item["class"] = local_map[original]
        for item in row.get("exclude", []):
            original = normalize(item["class"])
            item["class"] = local_map.get(original, fallback_map[original])
        row["prompt"] = rebuild_prompt(row, plural_by_class)
        output_rows.append(row)
    if replacement_index != len(replacements):
        raise RuntimeError(
            f"Prompt bank consumed {replacement_index:,}/{len(replacements):,} replacement classes"
        )
    return output_rows, assignments


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base_metadata)
    official_rows = read_jsonl(args.official_metadata)
    original_classes = read_original_classes(args.original_classes)
    alternate_classes = read_alternate_classes(args.alternate_classes)
    if len(original_classes) != len(alternate_classes):
        raise ValueError(
            f"Class counts differ: {len(original_classes)} original vs "
            f"{len(alternate_classes)} alternate"
        )
    if len(base_rows) != len(official_rows):
        raise ValueError(
            f"Prompt bank sizes differ: {len(base_rows)} base vs {len(official_rows)} official"
        )

    blocked = official_blocklist(official_rows)
    blocked.update(singular for singular, _ in alternate_classes)
    blocked.update(
        normalize(item["class"])
        for row in base_rows
        for item in row.get("include", [])
    )
    additional_classes = select_classes(imagenet21k_classes(), blocked, args.additional_classes)

    alternate_singulars = [singular for singular, _ in alternate_classes]
    alternate_plurals = dict(alternate_classes)
    fallback_map = dict(zip(original_classes, alternate_singulars, strict=True))
    class_slots_per_bank = sum(
        len({normalize(item["class"]) for item in row.get("include", [])})
        for row in official_rows
    )
    added_rows = []
    banks = []
    for bank_index, start in enumerate(range(0, len(additional_classes), class_slots_per_bank)):
        new_classes = additional_classes[start : start + class_slots_per_bank]
        plural_by_class = dict(alternate_plurals)
        plural_by_class.update({class_name: pluralize(class_name) for class_name in new_classes})
        bank_rows, assignments = remap_prompt_bank_by_occurrence(
            official_rows,
            new_classes,
            fallback_map,
            plural_by_class,
        )
        added_rows.extend(bank_rows)
        banks.append(
            {
                "bank_index": bank_index,
                "new_class_count": len(new_classes),
                "new_classes": new_classes,
                "assignments": assignments,
            }
        )
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
        "base_prompt_bank_count": 1,
        "additional_prompt_bank_count": len(banks),
        "prompts_per_bank": len(official_rows),
        "class_slots_per_bank": class_slots_per_bank,
        "prompt_source": str(args.official_metadata),
        "prompt_policy": "repeat_exact_geneval_structures_and_replace_class_slots_only",
        "chi_policy": "unchanged_and_supplied_by_exact_conditioning_exporter",
        "banks": banks,
        "official_class_overlap": leaked,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Base prompts: {len(base_rows):,}")
    print(f"Additional classes: {len(additional_classes):,}")
    print(f"Additional exact prompt banks: {len(banks):,}")
    print(f"Additional prompts: {len(added_rows):,}")
    print(f"Total fitting prompts: {len(combined_rows):,}")
    print(f"Total unique calculation classes: {len(combined_classes):,}")
    print(f"Official class overlap: {len(leaked)}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"First additional classes: {additional_classes[:10]}")


if __name__ == "__main__":
    main()
