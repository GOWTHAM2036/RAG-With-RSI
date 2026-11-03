"""
Automated Dataset Validator for RSI-RAG.
Validates processed JSONL datasets and evaluation splits against schema requirements.
Checks for missing questions, missing answers, empty contexts, duplicate IDs,
invalid records, and incorrect data types. Generates detailed JSON and console reports.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
from tabulate import tabulate

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import load_config, setup_logger

logger = setup_logger("validator", log_file="logs/validation.log")

class DatasetValidator:
    REQUIRED_KEYS = {"id", "question", "context", "answer", "source", "metadata"}

    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.storage_cfg = self.config.get("storage", {})
        self.processed_dir = Path(self.storage_cfg.get("processed_dir", "data/processed"))
        self.eval_dir = Path(self.storage_cfg.get("evaluation_dir", "data/evaluation"))

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validates a single JSONL dataset file.
        Returns a dictionary of detailed metrics and detected anomalies.
        """
        logger.info(f"Validating file: {file_path}...")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        total_records = 0
        invalid_json_count = 0
        missing_keys_count = 0
        missing_questions_count = 0
        missing_answers_count = 0
        empty_contexts_count = 0
        type_errors_count = 0
        duplicate_id_count = 0

        id_counter = Counter()
        sources_found = Counter()
        context_len_sum = 0
        question_len_sum = 0
        answer_len_sum = 0

        type_error_examples = []
        invalid_record_examples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                total_records += 1
                line = line.strip()
                if not line:
                    invalid_json_count += 1
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    invalid_json_count += 1
                    if len(invalid_record_examples) < 3:
                        invalid_record_examples.append(f"Line {line_idx}: JSON parse error: {e}")
                    continue

                if not isinstance(record, dict):
                    invalid_record_count += 1
                    continue

                # 1. Missing Required Keys Check
                missing_keys = self.REQUIRED_KEYS - set(record.keys())
                if missing_keys:
                    missing_keys_count += 1
                    if len(invalid_record_examples) < 3:
                        invalid_record_examples.append(f"Line {line_idx}: Missing keys: {list(missing_keys)}")
                    continue

                rec_id = record.get("id")
                question = record.get("question")
                context = record.get("context")
                answer = record.get("answer")
                source = record.get("source")
                metadata = record.get("metadata")

                # 2. Data Type Verification
                type_err = False
                if not isinstance(rec_id, str):
                    type_err = True
                if not isinstance(question, str):
                    type_err = True
                if not isinstance(context, list) or not all(isinstance(c, str) for c in context):
                    type_err = True
                if not isinstance(answer, str):
                    type_err = True
                if not isinstance(metadata, dict):
                    type_err = True

                if type_err:
                    type_errors_count += 1
                    if len(type_error_examples) < 3:
                        type_error_examples.append(f"Line {line_idx} (id={rec_id}): invalid data types")
                    continue

                # 3. Duplicate ID Check
                id_counter[rec_id] += 1
                sources_found[source] += 1

                # 4. Missing Questions Check
                if not question or not question.strip():
                    missing_questions_count += 1

                # 5. Missing Answers Check (Note: in SciFact retrieval-only evaluation or blind test, answer may legitimately be empty)
                if not answer or not answer.strip():
                    missing_answers_count += 1

                # 6. Empty Contexts Check
                if not context or all(len(c.strip()) == 0 for c in context):
                    empty_contexts_count += 1

                # Statistics accumulation
                question_len_sum += len(question.split()) if question else 0
                context_len_sum += len(context) if context else 0
                answer_len_sum += len(answer.split()) if answer else 0

        # Calculate duplicate IDs
        duplicate_id_count = sum(count - 1 for count in id_counter.values() if count > 1)
        valid_records = (
            total_records
            - invalid_json_count
            - missing_keys_count
            - type_errors_count
            - duplicate_id_count
            - missing_questions_count
        )

        n = max(1, total_records)
        report = {
            "file": str(file_path),
            "file_size_kb": round(file_path.stat().st_size / 1024, 2),
            "total_records": total_records,
            "valid_records": max(0, valid_records),
            "validity_percentage": round(max(0, valid_records) / n * 100, 2),
            "missing_questions": missing_questions_count,
            "missing_answers": missing_answers_count,
            "empty_contexts": empty_contexts_count,
            "duplicate_ids": duplicate_id_count,
            "invalid_json_or_records": invalid_json_count + missing_keys_count,
            "type_errors": type_errors_count,
            "sources_breakdown": dict(sources_found),
            "avg_question_words": round(question_len_sum / n, 2),
            "avg_context_documents": round(context_len_sum / n, 2),
            "avg_answer_words": round(answer_len_sum / n, 2),
            "notes": []
        }

        # Add notes clarifying benchmark design traits
        if "scifact" in file_path.name.lower() and missing_answers_count > 0:
            report["notes"].append(
                f"SciFact is a scientific document retrieval benchmark. Expected answers are intentionally empty ({missing_answers_count} records) without fake fabrication."
            )
        if "test_blind" in file_path.name.lower():
            report["notes"].append(
                "Blind test set: Expected answers are intentionally omitted to prevent data leakage during recursive self-improvement."
            )

        return report

    def run(self, dataset_name: Optional[str] = None, output_report: Optional[str] = None):
        """Runs validation across processed datasets and evaluation splits."""
        logger.info("=" * 70)
        logger.info("Starting RSI-RAG Dataset Validation")
        logger.info("=" * 70)

        files_to_check: List[Path] = []

        if dataset_name and dataset_name.lower() != "all":
            # Target specific dataset
            clean_name = dataset_name.lower().replace(".jsonl", "")
            target_p = self.processed_dir / f"{clean_name}.jsonl"
            if target_p.exists():
                files_to_check.append(target_p)
            else:
                # Check evaluation dir
                target_eval = self.eval_dir / f"{clean_name}.jsonl"
                if target_eval.exists():
                    files_to_check.append(target_eval)
                else:
                    raise FileNotFoundError(f"Could not find dataset file for: '{dataset_name}'")
        else:
            # Check all processed and evaluation jsonl files
            for p in sorted(self.processed_dir.glob("*.jsonl")):
                files_to_check.append(p)
            for p in sorted(self.eval_dir.glob("*.jsonl")):
                files_to_check.append(p)

        reports = []
        table_rows = []

        for f in files_to_check:
            rep = self.validate_file(f)
            reports.append(rep)

            table_rows.append([
                f.name,
                rep["total_records"],
                rep["valid_records"],
                f"{rep['validity_percentage']}%",
                rep["missing_questions"],
                rep["missing_answers"],
                rep["empty_contexts"],
                rep["duplicate_ids"],
                rep["type_errors"]
            ])

        # Print Tabular Summary to Console
        headers = [
            "Dataset File",
            "Total",
            "Valid",
            "Validity",
            "No Ques",
            "No Ans",
            "No Ctx",
            "Dup IDs",
            "Type Errs"
        ]
        table_str = tabulate(table_rows, headers=headers, tablefmt="grid")
        print("\n" + table_str + "\n")

        # Save Final JSON Report
        out_json_path = Path(output_report) if output_report else (self.processed_dir / "validation_report.json")
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        full_summary = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "total_files_validated": len(reports),
            "overall_records_examined": sum(r["total_records"] for r in reports),
            "reports": reports
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)

        # Also copy to evaluation directory for convenience
        eval_report_path = self.eval_dir / "validation_report.json"
        with open(eval_report_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Validation report successfully written to: {out_json_path}")
        logger.info(f"Validation report copy written to: {eval_report_path}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RSI-RAG Dataset Validator")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        help="Dataset name to validate (e.g. hotpotqa, ragbench, scifact, combined, or all)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        default=None,
        help="Custom path to output JSON validation report"
    )
    args = parser.parse_args()

    validator = DatasetValidator(config_path=args.config)
    validator.run(dataset_name=args.dataset, output_report=args.report)

if __name__ == "__main__":
    main()
