# LittleCurriculum Filter

A five-stage filtering pipeline for building **K–5 (kindergarten to 5th grade)** text corpora.

This is the pipeline used to build [LittleCurriculum](https://huggingface.co/datasets/littlelearner/LittleCurriculum), the ~88B-token K–5 corpus behind the LittleLearner models. The same pipeline was also used to select K–5 data for supervised fine-tuning (SFT) in our post-training experiments.

## Pipeline

LittleCurriculum is produced from FineWeb-Edu using five sequential stages:

1. Age-of-Acquisition and word-frequency pre-filtering
2. fastText grade-level classification
3. ModernBERT grade-level classification
4. Advanced mathematical and symbolic notation filtering
5. Frequency sampling based on Beyond-K–5-associated vocabulary

## Intended Use

Read this before applying the pipeline to data that differs substantially from web prose.

**The classifiers are trained for web prose.** Stages 2 and 3 were trained on labels generated from FineWeb-Edu documents. Performance may degrade on substantially different data distributions, in which case retraining the classifiers is recommended.

**The grade boundary is fixed to K–5.** Stages 2 and 3 use classifiers specifically trained to distinguish K–5 from higher-grade content. Retargeting the pipeline to a different grade boundary therefore requires retraining these classifiers.

**The classifiers expect whole documents.** They estimate the overall grade level of a document and have little context to work with for very short snippets. We recommend applying the pipeline to full documents rather than individual sentences.

## Install

```bash
pip install -r requirements.txt
python download_artifacts.py     # download classifier weights from the Hugging Face Hub
```

## Run

### Local files

```bash
python filter_k5.py --in shard.parquet --out kept.parquet
python filter_k5.py --in docs.jsonl --out kept.jsonl
```

Use `--text-field` if the document text is not stored under `text`.

### Hugging Face datasets

```bash
python filter_k5.py --hf-dataset HuggingFaceFW/fineweb-edu \
    --hf-config sample-10BT --out kept.parquet
```

Hub datasets are streamed and processed in batches. The `train` split is used.

### Test a configuration

```bash
python filter_k5.py --hf-dataset HuggingFaceFW/fineweb-edu \
    --limit 1000 --out probe.parquet
```

`--limit` works for both local files and Hugging Face datasets. Each run reports retention after every filtering stage.

## All options

| Option                 | Default    | Description                                 |
| ---------------------- | ---------- | ------------------------------------------- |
| `--in PATH`            | —          | Local `.parquet`, `.pq`, or `.jsonl` input. |
| `--hf-dataset REPO_ID` | —          | Hugging Face dataset to stream.             |
| `--hf-config`          | —          | Optional dataset configuration.             |
| `--out PATH`           | *required* | Output path.                                |
| `--text-field`         | `text`     | Field containing document text.             |
| `--batch-rows`         | `50000`    | Rows processed per batch for Parquet and Hub inputs.|
| `--limit N`            | —          | Stop after N input documents.               |


## Retraining the classifiers

Stages 2 and 3 are trained for the K–5 boundary, so retargeting to a different grade band means retraining them. The prompts used to produce the training labels are in `prompts/`.

Labels were generated with Gemini and kept only where all three prompt variants agreed on the grade band:

| File | Variant |
| --- | --- |
| `aggregated.txt` | Original, written against the Common Core State Standards |
| `aggregated_evolved_gemini-3-flash.txt` | Refined with OpenEvolve |
| `optimized_dspy_gemini-3-flash.txt` | Refined with DSPy |

Each assigns one of K5, K8, K12 or OOS. Adapt the grade bands, relabel a sample of your own corpus, and retrain.

## Acknowledgements

The Age-of-Acquisition norms used in the first filtering stage are from Kuperman, Stadthagen-Gonzalez, and Brysbaert (2012), *Age-of-acquisition ratings for 30,000 English words*, Behavior Research Methods 44(4).

The grade-level training labels were generated using Google Gemini, and the fastText and ModernBERT classifiers were trained on these labels.

We thank the authors and maintainers of FineWeb-Edu, the Age-of-Acquisition norms, fastText, and ModernBERT for making these resources available.

## License

Apache-2.0 for the code and classifier weights. The LittleCurriculum corpus itself is licensed under ODC-BY 1.0, inherited from FineWeb-Edu.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{li2026littlelearner,
      title={LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure},
      author={Fanfei Li and Jana Zeller and Manuel Prada-Corral and Thaddäus Wiedemer and Prasanna Mayilvahanan and Ryan Cotterell and Wieland Brendel},
      year={2026},
      eprint={2608.13545},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.13545},
}
```
