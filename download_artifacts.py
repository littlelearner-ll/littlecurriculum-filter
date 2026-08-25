#!/usr/bin/env python3
"""Fetch the filter artifacts (~358 MB) from the Hugging Face Hub.

    python download_artifacts.py

Creates:
    data/aoa.parquet              age-of-acquisition dictionary  (stage 1)
    data/word_log_odds.parquet    log-odds word statistics       (stage 5)
    models/fasttext_grade.bin     fastText grade classifier      (stage 2)
    models/modernbert_grade/      ModernBERT grade classifier    (stage 3)
"""
import argparse
from pathlib import Path

DEFAULT_REPO = "littlelearner/littlecurriculum-filter"
FILES = [
    "data/aoa.parquet",
    "data/word_log_odds.parquet",
    "models/fasttext_grade.bin",
    "models/modernbert_grade/config.json",
    "models/modernbert_grade/model.safetensors",
    "models/modernbert_grade/tokenizer.json",
    "models/modernbert_grade/tokenizer_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    for rel in FILES:
        print(f"  fetching {rel} ...", flush=True)
        hf_hub_download(repo_id=args.repo, filename=rel, revision=args.revision,
                        local_dir=str(args.dest))
    print(f"\nDone. Artifacts under {args.dest}")
    print("Hugging Face verifies file checksums on download.")


if __name__ == "__main__":
    main()
