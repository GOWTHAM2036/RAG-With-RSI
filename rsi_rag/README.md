# RSI-RAG: Automated Dataset Download & Preparation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Datasets](https://img.shields.io/badge/HuggingFace-datasets-yellow.svg)](https://huggingface.co/docs/datasets)
[![Tests Passing](https://img.shields.io/badge/pytest-7%20passed-brightgreen.svg)]()

Production-grade automated dataset downloading, preprocessing, validation, and evaluation split preparation pipeline for **Recursive Self-Improving Retrieval-Augmented Generation (RSI-RAG)** systems.

---

## 📁 1. Complete Project Structure

```
RSI-RAG/
├── data/
│   ├── raw/                           # Raw datasets saved via save_to_disk()
│   │   ├── hotpotqa/                  # HotpotQA raw Arrow table & metadata (5.89 MB)
│   │   ├── ragbench/                  # RAGBench raw Arrow table & metadata (3.37 MB)
│   │   └── scifact/                   # SciFact queries, corpus, qrels (0.59 MB)
│   │
│   ├── processed/                     # Common JSONL normalized format
│   │   ├── hotpotqa.jsonl             # 1,000 processed records (6.11 MB)
│   │   ├── ragbench.jsonl             # 600 processed records (1.56 MB)
│   │   ├── scifact.jsonl              # 200 processed records (0.44 MB)
│   │   ├── combined.jsonl             # 1,800 combined records (8.10 MB)
│   │   └── validation_report.json     # Machine-readable validation output
│   │
│   └── evaluation/                    # RSI-RAG isolated evaluation splits
│       ├── train.jsonl                # 1,260 records (70% split for training/optimization)
│       ├── validation.jsonl           # 270 records (15% split for tuning)
│       ├── test_isolated.jsonl        # 270 records (15% ground truth isolated evaluation vault)
│       ├── test_blind.jsonl           # 270 records (test queries/contexts with answers STRIPPED)
│       └── validation_report.json     # Copy of validation metrics
│
├── scripts/
│   ├── utils.py                       # Text normalization, retries, logger, config loader
│   ├── download_datasets.py           # Streamed dataset downloader with caching & retries
│   ├── preprocess_datasets.py         # Multi-dataset parser & evaluation split generator
│   └── validate_datasets.py           # Anomaly detector & tabular validation reporter
│
├── config/
│   └── dataset_config.yaml            # Declarative YAML configuration
│
├── tests/
│   └── test_pipeline.py               # Unit & integration tests for schemas and isolation
│
├── logs/
│   ├── download.log                   # Network download & saving trace
│   ├── preprocess.log                 # Preprocessing logs & record counts
│   └── validation.log                 # Anomaly detection & integrity log
│
├── requirements.txt                   # Dependency list
├── pytest.ini                         # Test runner configuration
└── README.md                          # Full documentation & usage guide
```

---

## 📚 2. Dataset Overview & HF Schemas

| Dataset | Hugging Face Source | Config / Splits | Purpose in RSI-RAG | Special Handling |
|---|---|---|---|---|
| **HotpotQA** | `hotpotqa/hotpot_qa` | `distractor`<br>*(train, validation)* | Multi-hop reasoning and distractor discrimination | Preserves 10 context articles (including distractor articles) and extracts supporting facts (`title` and `sent_id`). |
| **RAGBench** | `rungalileo/ragbench` | `hotpotqa` (or domain)<br>*(train, val, test)* | Retrieval quality, context relevance, faithfulness | Extracts candidate retrieval documents from `documents` list and reference answers from `response`. Preserves Trulens & RAGAS scores. |
| **SciFact** | `BeIR/scifact`<br>`BeIR/scifact-qrels` | `corpus`, `queries`<br>`default` (qrels) | Scientific document retrieval evaluation | Joins scientific claims (`queries`) with verified ground-truth evidence papers (`corpus`) using `qrels` mapping. Truthfully marks `answer: ""` without fabricating fake text. |

---

## ⚙️ 3. Installation

Activate your Python virtual environment (Python 3.10+):

```bash
cd "C:\Users\B Gowtham Reddy\RSI-RAG"
pip install -r requirements.txt
```

---

## 🚀 4. Execution Commands

### Full Pipeline Run
```bash
# 1. Download raw datasets with streaming limits
python scripts/download_datasets.py

# 2. Preprocess into unified JSONL & prepare evaluation splits
python scripts/preprocess_datasets.py

# 3. Validate schema integrity & generate report
python scripts/validate_datasets.py
```

### Single Dataset Execution
All scripts support selecting an individual dataset via the `--dataset` / `-d` argument:

```bash
# Download only HotpotQA with 100 samples
python scripts/download_datasets.py --dataset hotpotqa --sample-limit 100

# Preprocess only RAGBench
python scripts/preprocess_datasets.py --dataset ragbench

# Validate only SciFact
python scripts/validate_datasets.py --dataset scifact
```

### Running Test Suite
```bash
python -m pytest -v
```

---

## 📄 5. Sample Processed JSONL Records

### HotpotQA Record:
```json
{
  "id": "hotpotqa_5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "context": [
    "Arthur's Magazine: Arthur's Magazine (1844–1846) was an American literary magazine published in the 19th century.",
    "First for Women: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989."
  ],
  "answer": "Arthur's Magazine",
  "source": "hotpotqa",
  "metadata": {
    "original_id": "5a7a06935542990198eaf050",
    "split": "train",
    "type": "comparison",
    "level": "medium",
    "num_context_docs": 10,
    "supporting_facts": [
      {"title": "Arthur's Magazine", "sent_id": 0},
      {"title": "First for Women", "sent_id": 0}
    ]
  }
}
```

### RAGBench Record:
```json
{
  "id": "ragbench_5a732ab25542991f29ee2d25",
  "question": "In what school district is Governor John R. Rogers High School, named after John Rankin Rogers, located?",
  "context": [
    "Governor John R. Rogers High School is a high school in the Puyallup School District of Washington."
  ],
  "answer": "Governor John R. Rogers High School, named after John Rankin Rogers, is located in the Puyallup School District.",
  "source": "ragbench",
  "metadata": {
    "original_id": "5a732ab25542991f29ee2d25",
    "split": "train",
    "benchmark_domain": "hotpotqa",
    "ragas_faithfulness": 1.0,
    "relevance_score": 1.0,
    "num_context_docs": 4
  }
}
```

### SciFact Record:
```json
{
  "id": "scifact_1",
  "question": "0-dimensional biomaterials lack inductive properties.",
  "context": [
    "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging."
  ],
  "answer": "",
  "source": "scifact",
  "metadata": {
    "original_id": "1",
    "task_type": "scientific_retrieval_evaluation",
    "has_ground_truth_answer": false,
    "relevant_corpus_ids": ["31715818"],
    "num_relevant_docs": 1
  }
}
```

---

## 📊 6. Validation Results Table

Output generated by `scripts/validate_datasets.py`:

```
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| Dataset File        |   Total |   Valid | Validity   |   No Ques |   No Ans |   No Ctx |   Dup IDs |   Type Errs |
+=====================+=========+=========+============+===========+==========+==========+===========+=============+
| combined.jsonl      |    1800 |    1800 | 100.0%     |         0 |      200 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| hotpotqa.jsonl      |    1000 |    1000 | 100.0%     |         0 |        0 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| ragbench.jsonl      |     600 |     600 | 100.0%     |         0 |        0 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| scifact.jsonl       |     200 |     200 | 100.0%     |         0 |      200 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| test_blind.jsonl    |     270 |     270 | 100.0%     |         0 |      270 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| test_isolated.jsonl |     270 |     270 | 100.0%     |         0 |       30 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| train.jsonl         |    1260 |    1260 | 100.0%     |         0 |      140 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
| validation.jsonl    |     270 |     270 | 100.0%     |         0 |       30 |        0 |         0 |           0 |
+---------------------+---------+---------+------------+-----------+----------+----------+-----------+-------------+
```

---

## 💾 7. Actual Dataset Sizes and Disk Usage

| Stage | Path | Number of Records | Disk Size |
|---|---|---|---|
| **Raw HotpotQA** | `data/raw/hotpotqa/` | 1,000 samples (500 train, 500 val) | 5.89 MB |
| **Raw RAGBench** | `data/raw/ragbench/` | 600 samples (200 train, 200 val, 200 test) | 3.37 MB |
| **Raw SciFact** | `data/raw/scifact/` | 200 queries, 353 corpus docs, 1,258 qrels | 0.59 MB |
| **Processed JSONL** | `data/processed/` | 1,800 records across 4 files | 16.20 MB |
| **Evaluation Splits**| `data/evaluation/`| 2,070 records (train, val, test_iso, test_blind) | 9.25 MB |
| **Total Pipeline** | `RSI-RAG/` | 3,870 records | **~35.3 MB** |

---

## 🛡️ 8. RSI-RAG Recursive Evaluation Isolation

To prevent **data leakage and test contamination** during recursive self-improving search:

1. **`test_isolated.jsonl`**: The gold-standard test set containing expected answers. Stored strictly in an isolated evaluation vault.
2. **`test_blind.jsonl`**: Identical queries and candidate contexts, but **expected answers and grading rubrics are strictly stripped (`answer: ""`)**. The recursive self-improvement agent operates on this split to propose RAG mutations and prompt revisions without being able to memorize test answers.

---

## 🔍 9. Hugging Face Schema Incompatibilities & Solutions

1. **`rungalileo/ragbench` lacks a default configuration**:
   * *Problem*: Calling `load_dataset('rungalileo/ragbench')` raises `ValueError: Config name is missing`.
   * *Solution*: The pipeline automatically defaults to `'hotpotqa'` (or whatever sub-benchmark is declared in `dataset_config.yaml`).

2. **`BeIR/scifact` decoupled corpus and query architecture**:
   * *Problem*: `BeIR/scifact` does not provide monolithic QA pairs; it distributes documents into `corpus`, claims into `queries`, and relevance into `BeIR/scifact-qrels`.
   * *Solution*: The downloader fetches the queries, looks up corresponding ground-truth positive documents in `qrels`, downloads target corpus passages along with negative distractors, and joins them seamlessly during preprocessing.

3. **Missing answers in SciFact**:
   * *Problem*: SciFact evaluates document retrieval for scientific claims, not generative answer synthesis.
   * *Solution*: Rather than fabricating hallucinations, the pipeline leaves `answer: ""` and flags `has_ground_truth_answer: false` in metadata, verified accurately in the validation report.
