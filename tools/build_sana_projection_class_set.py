#!/usr/bin/env python3
"""Build a WordNet calculation dictionary while excluding GenEval classes.

This mirrors the class augmentation used in the local OV-DQUO experiments:
many semantic noun classes are used for calculating the projector, and a
bounded set of prompt templates is averaged later into one prototype per
class.  The official GenEval prompts and object labels are not calculation
rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable


# Same ordering as OV-DQUO/models/clip/prompts.py. The original OV-DQUO tools
# supported both a five-template calculation and a larger 70-template run;
# --num-templates may select up to all 80.
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-metadata",
        type=Path,
        default=Path("tools/metrics/geneval/prompts/evaluation_metadata.jsonl"),
    )
    parser.add_argument(
        "--candidate-classes",
        type=Path,
        help="Optional text file containing candidate classes. Otherwise NLTK WordNet is used.",
    )
    parser.add_argument("--num-classes", type=int, default=3_000)
    parser.add_argument("--num-templates", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/text_embeddings/sana_class_projection/spec"),
    )
    args = parser.parse_args()
    if args.num_classes <= 0:
        parser.error("--num-classes must be positive")
    if not 1 <= args.num_templates <= len(IMAGENET_TEMPLATES):
        parser.error(f"--num-templates must be between 1 and {len(IMAGENET_TEMPLATES)}")
    return args


def normalize_label(value: str) -> str:
    value = value.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", value)


def unique_in_order(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = normalize_label(value)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def load_benchmark_classes(path: Path) -> list[str]:
    classes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "include" not in row:
                raise ValueError(f"Missing include field at {path}:{line_number}")
            classes.extend(item["class"] for item in row["include"])
    return unique_in_order(classes)


def load_wordnet():
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required when --candidate-classes is omitted. Install only NLTK with "
            "`python -m pip install --no-deps nltk`, then run `python -m nltk.downloader wordnet`."
        ) from exc
    try:
        next(iter(wn.all_synsets(pos=wn.NOUN)))
    except LookupError as exc:
        raise RuntimeError("WordNet data is missing. Run `python -m nltk.downloader wordnet`.") from exc
    return wn


def benchmark_blocklist(benchmark_classes: list[str], wn=None) -> set[str]:
    blocked = set(benchmark_classes)
    if wn is None:
        return blocked
    for class_name in benchmark_classes:
        query = class_name.replace(" ", "_")
        for synset in wn.synsets(query, pos=wn.NOUN):
            blocked.update(normalize_label(lemma.name()) for lemma in synset.lemmas())
    return blocked


def wordnet_candidates(wn) -> list[str]:
    # This is the deterministic counterpart of the OV-DQUO WordNet noun pool:
    # the primary lemma from each noun synset, with duplicates removed in the
    # WordNet traversal order.
    return unique_in_order(synset.lemmas()[0].name() for synset in wn.all_synsets(pos=wn.NOUN))


def text_file_candidates(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return unique_in_order(line for line in handle if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    benchmark_classes = load_benchmark_classes(args.eval_metadata)
    wn = None if args.candidate_classes else load_wordnet()
    blocked = benchmark_blocklist(benchmark_classes, wn)
    candidates = (
        text_file_candidates(args.candidate_classes) if args.candidate_classes else wordnet_candidates(wn)
    )
    calculation_classes = [name for name in candidates if name not in blocked]
    if len(calculation_classes) < args.num_classes:
        raise ValueError(
            f"Only {len(calculation_classes):,} non-benchmark candidate classes are available; "
            f"requested {args.num_classes:,}"
        )
    calculation_classes = calculation_classes[: args.num_classes]
    templates = IMAGENET_TEMPLATES[: args.num_templates]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_path = output_dir / "calculation_classes.txt"
    spec_path = output_dir / "class_set.json"
    atomic_write_text(classes_path, "".join(f"{name}\n" for name in calculation_classes))

    spec = {
        "format_version": 1,
        "kind": "sana_projection_class_set",
        "class_source": str(args.candidate_classes) if args.candidate_classes else "nltk_wordnet_noun_synsets",
        "class_count": len(calculation_classes),
        "template_count": len(templates),
        "class_names": calculation_classes,
        "templates": templates,
        "benchmark_metadata": str(args.eval_metadata),
        "benchmark_metadata_sha256": sha256_file(args.eval_metadata),
        "excluded_benchmark_classes": benchmark_classes,
        "blocked_label_count": len(blocked),
    }
    atomic_write_text(spec_path, json.dumps(spec, indent=2, ensure_ascii=False) + "\n")

    print(f"Calculation classes: {len(calculation_classes):,}")
    print(f"Templates per class: {len(templates)}")
    print(f"Intermediate prompt strings: {len(calculation_classes) * len(templates):,}")
    print(f"Excluded GenEval object classes: {len(benchmark_classes)}")
    print(f"Saved class list: {classes_path}")
    print(f"Saved shared specification: {spec_path}")


if __name__ == "__main__":
    main()
