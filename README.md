# Session-Aware Recommender

A Python research pipeline for **session-aware product re-ranking**, combining learning-to-rank, neural recommendation models, and relevance–diversity evaluation.

I'm interested in how a shopper's recent interactions can improve recommendations—and how to evaluate those improvements without leaking future information. This project explores that question using Diginetica's supplied product candidates, with reproducible data preparation, model comparisons, and frozen evaluation protocols.

**Stack:** Python · LightGBM · PyTorch · RecBole · pandas · NumPy

## What it does

- Builds causal session histories from views, clicks, and purchases, using only interactions available before each query.
- Trains LambdaMART rankers with session, temporal, popularity, and collaborative features.
- Compares neural recommendation models and tree–neural score fusion against simpler baselines.
- Applies fixed and adaptive diversification policies to study the trade-off between relevance and product variety.
- Evaluates ranking quality, intra-list diversity, uncertainty, and inference costs, with data and protocol audits.

The main task is **re-ranking an existing candidate set**. It is distinct from full-catalog next-item retrieval, and results depend on the candidates and evaluation population.

## Pipeline

```text
Diginetica logs and candidates
            |
  Causal session histories
            |
 Features and temporal splits
            |
 Rankers and neural baselines
            |
 Re-ranking and diversification
            |
 Ranking metrics, audits, and reports
```

Model and policy selection use development/validation data. Reserved confirmation outcomes are evaluated only after choices have been frozen.

## Get started

Use **Python 3.11**. Allow about **5 GB** for the source archive, prepared data, and initial outputs; neural training and larger experiments require additional resources.

```bash
git clone https://github.com/ho-thanh-und/session-aware-recommender.git
cd session-aware-recommender
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`.

The dependency set includes PyTorch and RecBole. For a particular GPU setup, install the appropriate PyTorch build before installing this project. The tracked lock files record dependency versions for reference.

### Prepare the data

```bash
python scripts/download_data.py
python -m session_rerank.prepare
```

The downloader retrieves the public Diginetica archive and records source checksums. Raw data, checkpoints, and generated experiment outputs remain local and are excluded from Git.

### Start with the baseline

```bash
python -m session_rerank.benchmark --stage development
```

For the later session benchmark, run preparation and development before opening confirmation:

```bash
python -m session_rerank.benchmark_data
python -m session_rerank.benchmark_experiment development
python -m session_rerank.benchmark_experiment confirmation
python -m session_rerank.benchmark_experiment checks
python -m session_rerank.benchmark_experiment report
```

These stages can be computationally expensive. Confirmation is a reserved evaluation, not a tuning loop; preserve its records and use a separate checkout for new experiments. Older optimization workflows and their protocols are documented in the research notes.

## Findings and interpretation

The experiments investigate both stronger relevance models and diversification. The recorded analyses show that richer temporal/context features can improve ranking, while the tested adaptive diversity rules do not consistently outperform simpler static policies. A neural component does not automatically improve a strong tree model.

Detailed measurements and their evaluation populations are documented in:

- [Session benchmark status](EXPERIMENT_STATUS.md)
- [Earlier experiments and model comparisons](docs/README_FULL.md)
- [Optimization results](docs/OPTIMIZATION_RESULTS.md)
- [Literature notes and interpretation limits](LITERATURE_NOTES.md)

These documents cover different protocols and samples; their scores are not interchangeable. Generated result artifacts are not bundled with a fresh clone. Offline click-ranking and lexical diversity metrics do not establish sales lift, shopper satisfaction, fairness, or causal effects.

## Checks and reproducibility

```bash
python -m pip install pytest
python -m pytest
```

Tests cover ranking and re-ranking behavior, data preparation, and experiment contracts. Record protocol files, seeds, input checksums, package versions, and hardware alongside new results. Keep validation decisions separate from confirmation outcomes.

## Project layout

| Path | Purpose |
| :--- | :--- |
| `src/session_rerank/` | Data preparation, features, models, evaluation, and audits |
| `config/` | Experiment protocols and model settings |
| `scripts/` | Data download utilities |
| `tests/` | Regression and protocol checks |
| `docs/` | Methods, experiment history, and detailed findings |
| `results/shared/` | Location for compact, reviewed result artifacts |
| `requirements*.txt` | Recorded dependency versions |

## Contributors and data

Contributors: **Ho Chi Thanh, Nguyen Dinh Nam, and R Arulkumaran**. This is my personal portfolio copy of [the collaborative project](https://github.com/ho-thanh-und/cp4285-g02); the original repository retains its development history.

Dataset and third-party materials retain their respective terms. Source data is downloaded separately rather than redistributed here.
