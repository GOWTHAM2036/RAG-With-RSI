"""
Unit and integration tests for RSI-RAG dataset preparation pipeline.
"""

import os
import json
import pytest
from pathlib import Path

from scripts.utils import normalize_text, load_config
from scripts.validate_datasets import DatasetValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_text_normalization():
    raw_text = "  Hello \t\t world  \r\n This is   a   test. \n\n\n\n Paragraph 2.  "
    normalized = normalize_text(raw_text)
    assert "\t" not in normalized
    assert "\r" not in normalized
    assert "Paragraph 2." in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Hello world")

def test_config_loading():
    config = load_config(str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    assert "datasets" in config
    assert "hotpotqa" in config["datasets"]
    assert "ragbench" in config["datasets"]
    assert "scifact" in config["datasets"]
    assert config["sampling"]["seed"] == 42

def test_processed_hotpotqa_schema():
    p = PROJECT_ROOT / "data" / "processed" / "hotpotqa.jsonl"
    assert p.exists(), "hotpotqa.jsonl must exist"
    with open(p, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    required_keys = {"id", "question", "context", "answer", "source", "metadata"}
    assert required_keys.issubset(first_line.keys())
    assert isinstance(first_line["context"], list)
    assert len(first_line["context"]) > 0
    assert len(first_line["answer"]) > 0
    assert first_line["source"] == "hotpotqa"

def test_processed_ragbench_schema():
    p = PROJECT_ROOT / "data" / "processed" / "ragbench.jsonl"
    assert p.exists(), "ragbench.jsonl must exist"
    with open(p, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    required_keys = {"id", "question", "context", "answer", "source", "metadata"}
    assert required_keys.issubset(first_line.keys())
    assert isinstance(first_line["context"], list)
    assert len(first_line["answer"]) > 0
    assert first_line["source"] == "ragbench"

def test_processed_scifact_schema():
    p = PROJECT_ROOT / "data" / "processed" / "scifact.jsonl"
    assert p.exists(), "scifact.jsonl must exist"
    with open(p, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    required_keys = {"id", "question", "context", "answer", "source", "metadata"}
    assert required_keys.issubset(first_line.keys())
    assert isinstance(first_line["context"], list)
    assert first_line["source"] == "scifact"
    # Verify we did not fabricate an answer
    assert first_line["metadata"]["has_ground_truth_answer"] is False

def test_evaluation_splits_and_isolation():
    eval_dir = PROJECT_ROOT / "data" / "evaluation"
    train_p = eval_dir / "train.jsonl"
    val_p = eval_dir / "validation.jsonl"
    test_iso_p = eval_dir / "test_isolated.jsonl"
    test_blind_p = eval_dir / "test_blind.jsonl"

    assert train_p.exists()
    assert val_p.exists()
    assert test_iso_p.exists()
    assert test_blind_p.exists()

    # Read blind test line and verify answers are strictly omitted for recursive optimization safety
    with open(test_blind_p, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            assert r["answer"] == "", "Blind test records must not contain expected answers!"

    # Read isolated test line and verify answers exist for HotpotQA / RAGBench
    with open(test_iso_p, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        hotpot_in_test = [r for r in records if r["source"] == "hotpotqa"]
        assert len(hotpot_in_test) > 0
        assert len(hotpot_in_test[0]["answer"]) > 0

def test_validation_report_generation():
    validator = DatasetValidator(config_path=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    report = validator.validate_file(PROJECT_ROOT / "data" / "processed" / "combined.jsonl")

    assert report["total_records"] == 1800
    assert report["valid_records"] == 1800
    assert report["validity_percentage"] == 100.0
    assert report["duplicate_ids"] == 0
    assert report["missing_questions"] == 0
    assert report["type_errors"] == 0
