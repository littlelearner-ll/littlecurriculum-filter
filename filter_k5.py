"""Five-stage grade-level (K-5) text filter.

Usage:
    python filter_k5.py --in shard.parquet --out kept.parquet
    python filter_k5.py --in docs.jsonl    --out kept.jsonl --text-field content

Thresholds default to the published configuration.

See https://arxiv.org/abs/2608.13545 for more details. 
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE           = Path(__file__).resolve().parent
AOA_PARQUET    = HERE / "data" / "aoa.parquet"
COUNTS_PATH    = HERE / "data" / "word_log_odds.parquet"
FASTTEXT_MODEL = HERE / "models" / "fasttext_grade.bin"
MODERNBERT_DIR = HERE / "models" / "modernbert_grade"

# ---------------------------------------------------------------------------
# Filter constants
# ---------------------------------------------------------------------------
MIN_BEYOND_SCORE = 4.0
OOV_THRESHOLD   = 0.05
P95_MAX_AOA     = 12.0
ZIPF_SLOPE      = -1.504689
ZIPF_INTERCEPT  = 14.602716
TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")

# ---------------------------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------------------------
def load_aoa_dict() -> dict:
    aoa_df = pd.read_parquet(AOA_PARQUET, columns=["word", "aoa"])
    aoa_df["word"] = aoa_df["word"].str.lower().str.strip()
    aoa_df = aoa_df.dropna().drop_duplicates("word")
    return dict(zip(aoa_df["word"], aoa_df["aoa"]))


def load_blocklist(min_beyond_score: float = MIN_BEYOND_SCORE) -> frozenset:
    """Blocklist Beyond-K-5.
    """
    t = pq.read_table(COUNTS_PATH, columns=["word", "beyond_score"])
    words  = t.column("word").to_pylist()
    scores = t.column("beyond_score").to_numpy(zero_copy_only=False)
    return frozenset(w for w, sc in zip(words, scores)
                     if not np.isnan(sc) and sc >= min_beyond_score)


# ---------------------------------------------------------------------------
# Mask functions
# ---------------------------------------------------------------------------
def mask_aoa_zipf(texts: list[str], aoa_dict: dict,
                  max_aoa: float = P95_MAX_AOA,
                  max_oov: float = OOV_THRESHOLD) -> np.ndarray:
    import wordfreq
    zipf_cache: dict[str, float] = {}
    keep = []
    for txt in texts:
        tokens = TOKEN_RE.findall(txt.lower())
        if not tokens:
            keep.append(False)
            continue
        n_oov = 0
        aoa_vals = []
        for tok in tokens:
            if tok in aoa_dict:
                aoa_vals.append(aoa_dict[tok])
            else:
                z = zipf_cache.get(tok)
                if z is None:
                    z = wordfreq.zipf_frequency(tok, "en")
                    zipf_cache[tok] = z
                if z > 0:
                    aoa_vals.append(ZIPF_INTERCEPT + ZIPF_SLOPE * z)
                else:
                    n_oov += 1
        oov_frac = n_oov / len(tokens)
        if oov_frac > max_oov or not aoa_vals:
            keep.append(False)
            continue
        p95 = float(np.percentile(aoa_vals, 95))
        keep.append(p95 <= max_aoa)
    return np.array(keep)


def load_fasttext():
    """Load the stage-2 classifier."""
    import fasttext
    return fasttext.load_model(str(FASTTEXT_MODEL))


def load_modernbert():
    """Load the stage-3 classifier."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(MODERNBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODERNBERT_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer, device


def mask_fasttext(texts: list[str], model) -> np.ndarray:
    clean = [t.replace("\n", " ") for t in texts]
    labels, _ = model.predict(clean)
    return np.array([lbl[0] == "__label__K5" for lbl in labels])


def mask_modernbert(texts: list[str], model, tokenizer, device,
                    batch_size: int = 32) -> np.ndarray:
    import torch
    keep = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=2048,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        keep.extend(preds == 0)  # 0 = K5
    return np.array(keep)


SYMBOLIC_PATTERNS = [
    re.compile(r"[A-Za-z0-9][²³⁴⁵⁶⁷⁸⁹]"),
    re.compile(r"[A-Za-z0-9_)\]]\*\*\d+"),
    re.compile(r"\b\d{1,4}\s*\^\s*\d{1,3}\b"),
    re.compile(r"[A-Za-z]\s*\^\s*\{[^}]*\}"),
    re.compile(r"[A-Za-z0-9_)\]]\s*(?:\^|\*\*)\s*[-−]\s*\(?[\w\d]"),
    re.compile(r"[A-Za-z0-9_)\]]\s*(?:\^|\*\*)\s*\("),
    re.compile(r"[A-Za-z]\s*(?:\^|\*\*)\s*[A-Za-z]\b"),
    re.compile(r"∛"),
    re.compile(r"\bcbrt\s*\(", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9_)\]]\s*(?:\^|\*\*)\s*\(\s*1\s*/\s*3\s*\)"),
    re.compile(r"\\sqrt\[\s*[^\]]+\s*\]"),
    re.compile(r"[²³⁴⁵⁶⁷⁸⁹]\s*√"),
    re.compile(r"\bexp\s*\("),
    re.compile(r"\b(?:log|ln)\s*\("),
    re.compile(r"\b(?:arc)?(?:sin|cos|tan)h?\s*\(", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])[A-Za-z]\s*\^\s*2"),
    re.compile(r"√|\\sqrt"),
    re.compile(r"\b\d*[a-z]\s*[+\-]\s*\d*[a-z]\s*=", re.IGNORECASE),
    re.compile(r"[a-z]\s*\^\s*[3-9n]", re.IGNORECASE),
    re.compile(r"[a-z]\s*\^\s*2\s*[+\-]\s*\d*[a-z]", re.IGNORECASE),
    re.compile(r"[∑Σ]"),
    re.compile(r"[∫]"),
    re.compile(r"[∂]"),
    re.compile(r"\\(?:frac|lim|int|sum|partial|infty|sqrt\[)\b"),
    re.compile(r"[a-z0-9]\s*[<>≤≥]=?\s*[a-z]\s*[<>≤≥]=?\s*[a-z0-9]", re.IGNORECASE),
    re.compile(r"\b[fgh]\s*\(\s*[xyz]\s*\)"),
    re.compile(r"\b\d+\s*[xyz]\s*[+\-*/]\s*\d+\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"[A-Z][a-z]?\d+[A-Z]|[A-Z][a-z]?\d+\)"),  # chemical formulas
]


def mask_symbolic(texts: list[str]) -> np.ndarray:
    return np.array([not any(p.search(t) for p in SYMBOLIC_PATTERNS) for t in texts])


def mask_wordlist(texts: list[str], blocklist: frozenset) -> np.ndarray:
    keep = []
    for txt in texts:
        # Normalize case before blocklist matching.
        tokens = set(TOKEN_RE.findall(txt.lower()))
        keep.append(tokens.isdisjoint(blocklist))
    return np.array(keep)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def get_text(rec, text_field):
    if not isinstance(rec, dict):
        raise SystemExit(
            "each JSONL line must be a JSON object, e.g. "
            f'{{"{text_field}": "..."}} -- got {type(rec).__name__}')
    if text_field not in rec:
        raise SystemExit(
            f"field {text_field!r} not found in a record. "
            f"Available fields: {sorted(rec)}. Use --text-field.")
    return str(rec.get(text_field) or "").strip()


EXTENSIONS = {".parquet": "parquet", ".pq": "parquet", ".jsonl": "jsonl"}


def detect_format(path: Path, override: str | None) -> str:
    if override and override != "auto":
        return override
    suf = path.suffix.lower()
    if suf not in EXTENSIONS:
        raise SystemExit(
            f"cannot infer the input format from {path.name!r}. "
            f"Expected one of {', '.join(sorted(EXTENSIONS))}; "
            f"pass --format explicitly to override.")
    return EXTENSIONS[suf]


def read_jsonl(path: Path, text_field: str, limit: int | None = None):
    records, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            texts.append(get_text(rec, text_field))
            if limit and len(records) >= limit:
                break
    return records, texts


def iter_hf_batches(repo: str, config: str | None,
                    text_field: str, batch_rows: int, limit: int | None):
    """Stream a Hugging Face dataset and yield batches of rows and texts."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "--hf-dataset needs the datasets library: pip install datasets")
        
    ds = load_dataset(repo, config, split="train", streaming=True)
    rows, texts, seen = [], [], 0
    for rec in ds:
        if text_field not in rec:
            raise SystemExit(
                f"field {text_field!r} not in this dataset. "
                f"Available: {sorted(rec)}. Use --text-field.")
        rows.append(dict(rec))
        texts.append(str(rec.get(text_field) or "").strip())
        seen += 1
        if len(rows) >= batch_rows:
            yield rows, texts
            rows, texts = [], []
        if limit and seen >= limit:
            break
    if rows:
        yield rows, texts


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def build_mask(texts, args, aoa_dict, blocklist, ft_model,
               mb_model, mb_tokenizer, mb_device, verbose=True):
    n0 = len(texts)
    keep = np.ones(n0, dtype=bool)
    if n0 == 0:
        return keep

    def step(name, fn):
        nonlocal keep
        live = np.nonzero(keep)[0]
        if len(live) == 0:
            if verbose:
                print(f"  {name:<14s}: nothing left", flush=True)
            return
        t0 = time.time()
        m_live = fn([texts[i] for i in live])
        before = int(keep.sum())
        keep[live[~np.asarray(m_live)]] = False
        after = int(keep.sum())
        if verbose:
            print(f"  {name:<14s}: {after:>8,} / {n0:,} ({100*after/n0:5.2f}%) "
                  f"-- dropped {before-after:,}  [{time.time()-t0:.1f}s]", flush=True)

    step("1 AoA+Zipf",   lambda T: mask_aoa_zipf(T, aoa_dict, args.max_aoa, args.max_oov))
    step("2 FastText",   lambda T: mask_fasttext(T, ft_model))
    step("3 ModernBERT", lambda T: mask_modernbert(T, mb_model, mb_tokenizer, mb_device))
    step("4 Symbolic",   mask_symbolic)
    step("5 Wordlist",   lambda T: mask_wordlist(T, blocklist))
    return keep


def main():
    ap = argparse.ArgumentParser(
        description="Five-stage K5 grade-level text filter.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="inp", type=Path,
                     help="local .parquet / .pq / .jsonl file")
    src.add_argument("--hf-dataset", metavar="REPO_ID",
                     help="stream a dataset from the Hugging Face Hub instead "
                          "of reading a local file, e.g. "
                          "HuggingFaceFW/fineweb-edu")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--format", choices=["auto", "jsonl", "parquet"], default="auto",
                    help="input format (default: from the file extension)")
    ap.add_argument("--text-field", default="text",
                    help="column/field holding the "
                         "document text (default: text)")
    ap.add_argument("--hf-config", default=None,
                    help="dataset config name (e.g. sample-10BT)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many input documents; useful for pilot runs")
    ap.add_argument("--batch-rows", type=int, default=50000,
                    help="rows processed per batch (default: 50000)")
    # thresholds
    ap.add_argument("--max-aoa", type=float, default=P95_MAX_AOA,
                    help=f"max 95th-pct token age-of-acquisition (default: {P95_MAX_AOA})")
    ap.add_argument("--max-oov", type=float, default=OOV_THRESHOLD,
                    help=f"max out-of-vocabulary fraction (default: {OOV_THRESHOLD})")
    ap.add_argument("--min-beyond-score", type=float, default=MIN_BEYOND_SCORE,
                    help="stage-5 cutoff: block words whose Beyond-K-5 "
                         "association score log2(rate_beyond/rate_k5) is at "
                         f"or above this (default: {MIN_BEYOND_SCORE}; higher "
                         "blocks fewer words)")
    args = ap.parse_args()

    fmt = "hf" if args.hf_dataset else detect_format(args.inp, args.format)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "hf":
        cfg = f" [{args.hf_config}]" if args.hf_config else ""
        print(f"input : {args.hf_dataset}{cfg} "
              f"(streamed from the Hugging Face Hub)", flush=True)
    else:
        print(f"input : {args.inp}  (format: {fmt})", flush=True)
    print(f"config: max_aoa={args.max_aoa} max_oov={args.max_oov} "
          f"min_beyond_score={args.min_beyond_score}", flush=True)

    aoa_dict = load_aoa_dict()
    blocklist = load_blocklist(args.min_beyond_score)
    print(f"  AoA vocab: {len(aoa_dict):,} words", flush=True)
    print(f"  blocklist: {len(blocklist):,} words "
          f"(beyond-K5 score >= {args.min_beyond_score})", flush=True)

    # Load classifiers once for all batches.
    ft_model = load_fasttext()
    mb_model, mb_tokenizer, mb_device = load_modernbert()
    print(f"  classifiers loaded (ModernBERT on {mb_device})", flush=True)

    n_in = n_kept = 0
    t_start = time.time()

    if fmt == "hf":
        writer = None
        try:
            for rows, texts in iter_hf_batches(
                    args.hf_dataset, args.hf_config,
                    args.text_field, args.batch_rows, args.limit):
                print(f"\n--- rows {n_in:,}-{n_in+len(texts):,} ---", flush=True)
                keep = build_mask(texts, args, aoa_dict, blocklist, ft_model,
                                  mb_model, mb_tokenizer, mb_device)
                tbl = pa.Table.from_pylist([r for r, k in zip(rows, keep) if k])
                if len(tbl):
                    if writer is None:
                        writer = pq.ParquetWriter(args.out, tbl.schema,
                                                  compression="zstd")
                    writer.write_table(tbl)
                n_in += len(texts)
                n_kept += int(keep.sum())
        finally:
            if writer is not None:
                writer.close()
    elif fmt == "parquet":
        pf = pq.ParquetFile(args.inp)
        if args.text_field not in pf.schema_arrow.names:
            raise SystemExit(
                f"column {args.text_field!r} not in {args.inp}. "
                f"Available: {pf.schema_arrow.names}. Use --text-field.")
        writer = None
        try:
            for batch in pf.iter_batches(batch_size=args.batch_rows):
                tbl = pa.Table.from_batches([batch])
                if args.limit:
                    remaining = args.limit - n_in
                    if remaining <= 0:
                        break
                    if tbl.num_rows > remaining:
                        tbl = tbl.slice(0, remaining)
                texts = [str(t) if t is not None else ""
                         for t in tbl.column(args.text_field).to_pylist()]
                print(f"\n--- rows {n_in:,}-{n_in+len(texts):,} ---", flush=True)
                keep = build_mask(texts, args, aoa_dict, blocklist, ft_model,
                                  mb_model, mb_tokenizer, mb_device)
                out_tbl = tbl.filter(pa.array(keep.tolist()))
                if writer is None:
                    writer = pq.ParquetWriter(args.out, tbl.schema, compression="zstd")
                writer.write_table(out_tbl)
                n_in += len(texts)
                n_kept += int(keep.sum())
        finally:
            if writer is not None:
                writer.close()
    else:
        records, texts = read_jsonl(args.inp, args.text_field, args.limit)
        n_in = len(records)
        print(f"loaded {n_in:,} rows\n", flush=True)
        keep = build_mask(texts, args, aoa_dict, blocklist, ft_model,
                          mb_model, mb_tokenizer, mb_device)
        n_kept = int(keep.sum())
        with open(args.out, "w", encoding="utf-8") as out_f:
            for i, rec in enumerate(records):
                if keep[i]:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== summary ===")
    print(f"input:   {n_in:,}")
    print(f"kept:    {n_kept:,}  ({100*n_kept/max(n_in,1):.2f}%)")
    print(f"dropped: {n_in-n_kept:,}")
    print(f"output:  {args.out}")
    print(f"time:    {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
