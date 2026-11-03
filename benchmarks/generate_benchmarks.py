"""
Synthetic QA Benchmark Generator utility.
Extracts representative question-context-answer pairs from documents to bootstrap evaluation sets.
"""
from typing import List
import json
import re
from pathlib import Path
from backend.app.models import BenchmarkDataset, BenchmarkPair, Document
from backend.app.ingestion.loader import DocumentLoader

class BenchmarkGenerator:
    @staticmethod
    def generate_from_document(doc: Document) -> List[BenchmarkPair]:
        pairs = []
        # Split on sections
        sections = re.split(r"\n#{1,4}\s+", doc.content)
        for idx, sec in enumerate(sections):
            lines = sec.strip().split("\n")
            if not lines:
                continue
            heading = lines[0].strip().rstrip(":")
            body = " ".join([l.strip() for l in lines[1:] if l.strip()])
            if len(body) < 50:
                continue

            query = f"What are the key points regarding {heading}?"
            pairs.append(
                BenchmarkPair(
                    id=f"synth-{doc.id[:6]}-{idx}",
                    query=query,
                    ground_truth_contexts=[body[:300]],
                    ground_truth_answer=body[:200],
                    category="synthetic"
                )
            )
        return pairs

    @classmethod
    def save_benchmark(cls, dataset: BenchmarkDataset, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2)

if __name__ == "__main__":
    import sys
    docs_dir = Path(__file__).resolve().parent.parent / "backend" / "data" / "sample_docs"
    docs = DocumentLoader.load_directory(docs_dir)
    all_pairs = []
    for d in docs:
        all_pairs.extend(BenchmarkGenerator.generate_from_document(d))

    dataset = BenchmarkDataset(
        name="Auto-Generated Synthetic Corpus Benchmark",
        description="Bootstrap dataset derived from ingested sample files.",
        pairs=all_pairs
    )
    out_file = Path(__file__).resolve().parent / "synthetic_qa.json"
    BenchmarkGenerator.save_benchmark(dataset, out_file)
    print(f"Generated {len(all_pairs)} benchmark pairs to {out_file}")
