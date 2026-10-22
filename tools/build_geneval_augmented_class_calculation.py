#!/usr/bin/env python3
"""Build a 20,000-class SANA calculation set from the native Gemma vocabulary.

All 80 official GenEval object classes are intentionally included. The remaining
slots are filled with usable decoded surfaces from the native SANA/Gemma
tokenizer, in token-ID order. Selected classes fill the object slots of repeated
official 553-row GenEval banks. Only object class names (plus article/plural
agreement) are changed; tags, task structure, attributes, counts, relations, and
the CHI prefix used by the exact-conditioning exporter remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_OFFICIAL_METADATA = Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl")
DEFAULT_ORIGINAL_CLASSES = Path("tools/metrics/geneval/evaluation/object_names.txt")
DEFAULT_MODEL_ID = "Efficient-Large-Model/gemma-2-2b-it"
DEFAULT_TARGET_CLASS_COUNT = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-metadata", type=Path, default=DEFAULT_OFFICIAL_METADATA)
    parser.add_argument("--original-classes", type=Path, default=DEFAULT_ORIGINAL_CLASSES)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--target-classes", type=int, default=DEFAULT_TARGET_CLASS_COUNT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/text_embeddings/geneval_vocab20000"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.target_classes <= 0:
        parser.error("--target-classes must be positive")
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


def valid_visual_label(label: str) -> bool:
    return (
        2 <= len(label) <= 40
        and len(label.split()) <= 3
        and re.fullmatch(r"[a-z][a-z -]*", label) is not None
    )


def invalid_vocabulary_surface(surface: str) -> str | None:
    if not surface:
        return "empty_surface"
    if "\ufffd" in surface:
        return "unicode_replacement_character"
    if any(unicodedata.category(character) == "Cc" for character in surface):
        return "control_character"
    if not valid_visual_label(normalize(surface)):
        return "not_class_like"
    return None


def select_vocabulary_classes(
    tokenizer: Any,
    official_classes: list[str],
    target_count: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    if len(official_classes) > target_count:
        raise ValueError(
            f"The {len(official_classes)} official classes exceed the "
            f"requested total of {target_count}"
        )

    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    vocabulary_by_id: dict[int, str] = {}
    for token_string, token_id in tokenizer.get_vocab().items():
        token_id = int(token_id)
        vocabulary_by_id.setdefault(token_id, token_string)

    selected = list(official_classes)
    seen = set(selected)
    records = [
        {
            "class_name": class_name,
            "selection_source": "official_geneval",
            "token_ids": tokenizer.encode(class_name, add_special_tokens=False),
        }
        for class_name in official_classes
    ]
    exclusions: Counter[str] = Counter()

    for token_id in sorted(vocabulary_by_id):
        if len(selected) == target_count:
            break
        if token_id in special_ids:
            exclusions["special_token"] += 1
            continue

        surface = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        invalid_reason = invalid_vocabulary_surface(surface)
        if invalid_reason is not None:
            exclusions[invalid_reason] += 1
            continue

        class_name = normalize(surface)
        if class_name in seen:
            exclusions["duplicate_or_official_surface"] += 1
            continue

        prompt_context_ids = tokenizer.encode(
            f"a photo of a {class_name}",
            add_special_tokens=False,
        )
        if token_id not in prompt_context_ids:
            exclusions["token_not_preserved_in_prompt_context"] += 1
            continue

        selected.append(class_name)
        seen.add(class_name)
        records.append(
            {
                "class_name": class_name,
                "selection_source": "gemma_vocab_rank",
                "token_id": token_id,
                "token_string": vocabulary_by_id[token_id],
                "decoded_surface": surface,
                "prompt_context_token_ids": prompt_context_ids,
            }
        )

    if len(selected) == target_count:
        return selected, records, dict(exclusions)
    raise ValueError(
        f"Only {len(selected):,} usable unique classes were available from the "
        f"tokenizer; requested {target_count:,}"
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
    official_rows = read_jsonl(args.official_metadata)
    original_classes = read_original_classes(args.original_classes)
    official_classes = {
        normalize(item["class"])
        for row in official_rows
        for item in row.get("include", [])
    }
    if set(original_classes) != official_classes:
        raise ValueError(
            "The official object-name file does not match the classes in the "
            "official GenEval metadata"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    calculation_classes, selection_records, exclusion_counts = select_vocabulary_classes(
        tokenizer,
        original_classes,
        args.target_classes,
    )
    vocabulary_classes = calculation_classes[len(original_classes) :]

    fallback_map = {class_name: class_name for class_name in original_classes}
    official_plurals = {class_name: pluralize(class_name) for class_name in original_classes}
    class_slots_per_bank = sum(
        len({normalize(item["class"]) for item in row.get("include", [])})
        for row in official_rows
    )
    added_rows = []
    banks = []
    for bank_index, start in enumerate(range(0, len(vocabulary_classes), class_slots_per_bank)):
        new_classes = vocabulary_classes[start : start + class_slots_per_bank]
        plural_by_class = dict(official_plurals)
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
    combined_rows = official_rows + added_rows

    combined_classes = {
        normalize(item["class"])
        for row in combined_rows
        for item in row.get("include", [])
    }
    expected_classes = set(calculation_classes)
    if combined_classes != expected_classes:
        missing = sorted(expected_classes.difference(combined_classes))
        unexpected = sorted(combined_classes.difference(expected_classes))
        raise RuntimeError(
            f"Generated class set differs from selection; missing={missing}, "
            f"unexpected={unexpected}"
        )
    official_overlap = sorted(official_classes.intersection(combined_classes))
    if len(official_overlap) != len(original_classes):
        raise RuntimeError("Not all official GenEval classes are present in the calculation set")

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
        "kind": "geneval_native_vocab_calculation_set",
        "class_source": f"{args.model_id} tokenizer vocabulary",
        "class_selection": (
            "all_official_geneval_classes_then_lowest_rank_usable_unique_"
            "native_gemma_token_surfaces"
        ),
        "target_class_count": args.target_classes,
        "target_model": args.model_id,
        "tokenizer_vocab_size": len(tokenizer.get_vocab()),
        "official_metadata": str(args.official_metadata),
        "base_prompt_count": len(official_rows),
        "additional_prompt_count": len(added_rows),
        "total_prompt_count": len(combined_rows),
        "base_unique_class_count": len(original_classes),
        "vocabulary_class_count": len(vocabulary_classes),
        "total_unique_class_count": len(combined_classes),
        "calculation_classes": calculation_classes,
        "vocabulary_classes": vocabulary_classes,
        "selection_records": selection_records,
        "selection_exclusion_counts": exclusion_counts,
        "base_prompt_bank_count": 1,
        "additional_prompt_bank_count": len(banks),
        "prompts_per_bank": len(official_rows),
        "class_slots_per_bank": class_slots_per_bank,
        "prompt_source": str(args.official_metadata),
        "prompt_policy": "repeat_exact_geneval_structures_and_replace_class_slots_only",
        "chi_policy": "unchanged_and_supplied_by_exact_conditioning_exporter",
        "banks": banks,
        "evaluation_classes_intentionally_included": True,
        "seen_unseen_split": False,
        "official_class_overlap": official_overlap,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Official GenEval classes: {len(original_classes):,}")
    print(f"Gemma vocabulary classes: {len(vocabulary_classes):,}")
    print(f"Total unique calculation classes: {len(combined_classes):,}")
    print(f"Additional exact prompt banks: {len(banks):,}")
    print(f"Additional prompts: {len(added_rows):,}")
    print(f"Total fitting prompts: {len(combined_rows):,}")
    print(f"Official class overlap: {len(official_overlap)} (intentional)")
    print(f"Selection exclusions: {exclusion_counts}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")
    print(f"First vocabulary classes: {vocabulary_classes[:10]}")


if __name__ == "__main__":
    main()
