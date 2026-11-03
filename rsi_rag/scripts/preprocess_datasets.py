"""
Automated Dataset Preprocessor for RSI-RAG.
Normalizes HotpotQA, RAGBench, and SciFact into a unified, robust JSONL format.
Deduplicates records, preserves original IDs, supporting facts, and distractor context.
Prepares isolated evaluation splits (train, validation, test_isolated, test_blind).
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from datasets import load_from_disk
from tqdm import tqdm

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import load_config, setup_logger, normalize_text

logger = setup_logger("preprocessor", log_file="logs/preprocess.log")

class DatasetPreprocessor:
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        self.config = load_config(config_path)
        self.storage_cfg = self.config.get("storage", {})
        self.raw_dir = Path(self.storage_cfg.get("raw_dir", "data/raw"))
        self.processed_dir = Path(self.storage_cfg.get("processed_dir", "data/processed"))
        self.eval_dir = Path(self.storage_cfg.get("evaluation_dir", "data/evaluation"))
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.seed = self.config.get("sampling", {}).get("seed", 42)
        random.seed(self.seed)

    def preprocess_hotpotqa(self) -> List[Dict[str, Any]]:
        """
        Preprocesses raw HotpotQA dataset.
        Format in HotpotQA:
          id: string
          question: string
          answer: string
          context: {'title': List[str], 'sentences': List[List[str]]}
          supporting_facts: {'title': List[str], 'sent_id': List[int]}
        """
        src_path = self.raw_dir / "hotpotqa"
        if not src_path.exists():
            raise FileNotFoundError(f"Raw HotpotQA not found at {src_path}. Run download_datasets.py first.")

        logger.info(f"[HotpotQA] Loading raw dataset from {src_path}...")
        ds_dict = load_from_disk(src_path)
        processed_records = []
        seen_ids: Set[str] = set()

        for split_name in ds_dict.keys():
            split_ds = ds_dict[split_name]
            logger.info(f"[HotpotQA] Processing split '{split_name}' ({len(split_ds)} raw records)...")

            for item in tqdm(split_ds, desc=f"Preprocess HotpotQA [{split_name}]"):
                raw_id = item.get("id", "")
                unique_id = f"hotpotqa_{raw_id}" if raw_id else f"hotpotqa_{len(processed_records)}"

                if unique_id in seen_ids:
                    continue
                seen_ids.add(unique_id)

                question = normalize_text(item.get("question", ""))
                answer = normalize_text(item.get("answer", ""))

                # Reconstruct documents from title and sentences list
                raw_ctx = item.get("context", {})
                titles = raw_ctx.get("title", [])
                sentences_list = raw_ctx.get("sentences", [])

                context_docs = []
                for title, sents in zip(titles, sentences_list):
                    doc_title = normalize_text(title)
                    doc_body = " ".join([normalize_text(s) for s in sents if normalize_text(s)])
                    if doc_title and doc_body:
                        context_docs.append(f"{doc_title}: {doc_body}")
                    elif doc_body:
                        context_docs.append(doc_body)

                # Supporting evidence metadata
                raw_sup = item.get("supporting_facts", {})
                sup_titles = raw_sup.get("title", [])
                sup_sent_ids = raw_sup.get("sent_id", [])

                metadata = {
                    "original_id": raw_id,
                    "split": split_name,
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "num_context_docs": len(context_docs),
                    "supporting_facts": [
                        {"title": t, "sent_id": sid}
                        for t, sid in zip(sup_titles, sup_sent_ids)
                    ]
                }

                record = {
                    "id": unique_id,
                    "question": question,
                    "context": context_docs,
                    "answer": answer,
                    "source": "hotpotqa",
                    "metadata": metadata
                }
                processed_records.append(record)

        logger.info(f"[HotpotQA] Successfully preprocessed {len(processed_records)} valid records.")
        return processed_records

    def preprocess_ragbench(self) -> List[Dict[str, Any]]:
        """
        Preprocesses raw RAGBench dataset.
        Format in RAGBench:
          id: string
          question: string
          documents: List[str]
          response: string (ground truth / model response)
          evaluation scores: ragas_faithfulness, relevance_score, etc.
        """
        src_path = self.raw_dir / "ragbench"
        if not src_path.exists():
            raise FileNotFoundError(f"Raw RAGBench not found at {src_path}. Run download_datasets.py first.")

        logger.info(f"[RAGBench] Loading raw dataset from {src_path}...")
        ds_dict = load_from_disk(src_path)
        processed_records = []
        seen_ids: Set[str] = set()

        for split_name in ds_dict.keys():
            split_ds = ds_dict[split_name]
            logger.info(f"[RAGBench] Processing split '{split_name}' ({len(split_ds)} raw records)...")

            for item in tqdm(split_ds, desc=f"Preprocess RAGBench [{split_name}]"):
                raw_id = item.get("id", "")
                unique_id = f"ragbench_{raw_id}" if raw_id else f"ragbench_{len(processed_records)}"

                if unique_id in seen_ids:
                    continue
                seen_ids.add(unique_id)

                question = normalize_text(item.get("question", ""))
                # In RAGBench, 'response' contains the answer/reference text
                answer = normalize_text(item.get("response", ""))

                # Documents is a list of candidate documents
                raw_docs = item.get("documents", [])
                context_docs = [normalize_text(d) for d in raw_docs if normalize_text(d)]

                metadata = {
                    "original_id": raw_id,
                    "split": split_name,
                    "benchmark_domain": item.get("dataset_name", "unknown"),
                    "generation_model": item.get("generation_model_name", "unknown"),
                    "ragas_faithfulness": item.get("ragas_faithfulness"),
                    "ragas_context_relevance": item.get("ragas_context_relevance"),
                    "relevance_score": item.get("relevance_score"),
                    "all_relevant_sentence_keys": item.get("all_relevant_sentence_keys", []),
                    "num_context_docs": len(context_docs)
                }

                record = {
                    "id": unique_id,
                    "question": question,
                    "context": context_docs,
                    "answer": answer,
                    "source": "ragbench",
                    "metadata": metadata
                }
                processed_records.append(record)

        logger.info(f"[RAGBench] Successfully preprocessed {len(processed_records)} valid records.")
        return processed_records

    def preprocess_scifact(self) -> List[Dict[str, Any]]:
        """
        Preprocesses raw BeIR/scifact dataset (queries, corpus, qrels).
        Format:
          queries: _id, title, text
          corpus: _id, title, text
          qrels: query-id, corpus-id, score
        """
        src_path = self.raw_dir / "scifact"
        if not src_path.exists():
            raise FileNotFoundError(f"Raw SciFact not found at {src_path}. Run download_datasets.py first.")

        logger.info(f"[SciFact] Loading raw dataset from {src_path}...")
        ds_dict = load_from_disk(src_path)

        queries_ds = ds_dict.get("queries", [])
        corpus_ds = ds_dict.get("corpus", [])
        qrels_ds = ds_dict.get("qrels", [])

        # Build corpus lookup map: corpus_id -> text
        logger.info(f"[SciFact] Indexing {len(corpus_ds)} corpus documents...")
        corpus_map = {}
        for c in corpus_ds:
            cid = str(c.get("_id", ""))
            title = normalize_text(c.get("title", ""))
            body = normalize_text(c.get("text", ""))
            doc_str = f"{title}: {body}" if title and body else (body or title)
            if cid:
                corpus_map[cid] = doc_str

        # Build qrels mapping: query_id -> List[corpus_id]
        logger.info(f"[SciFact] Indexing {len(qrels_ds)} qrels mappings...")
        qrels_map: Dict[str, List[Dict[str, Any]]] = {}
        for qr in qrels_ds:
            qid = str(qr.get("query-id", ""))
            cid = str(qr.get("corpus-id", ""))
            score = qr.get("score", 1)
            if qid not in qrels_map:
                qrels_map[qid] = []
            qrels_map[qid].append({"corpus_id": cid, "score": score})

        # Assemble unified records
        processed_records = []
        seen_ids: Set[str] = set()

        logger.info(f"[SciFact] Processing {len(queries_ds)} queries...")
        for q in tqdm(queries_ds, desc="Preprocess SciFact [queries]"):
            raw_id = str(q.get("_id", ""))
            unique_id = f"scifact_{raw_id}" if raw_id else f"scifact_{len(processed_records)}"

            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)

            # In SciFact, 'text' is the scientific claim
            question = normalize_text(q.get("text", q.get("title", "")))

            # Retrieve supporting context from corpus using qrels
            relevant_hits = qrels_map.get(raw_id, [])
            context_docs = []
            rel_cids = []

            for hit in relevant_hits:
                cid = hit["corpus_id"]
                rel_cids.append(cid)
                if cid in corpus_map:
                    context_docs.append(corpus_map[cid])

            # In retrieval evaluation benchmarks, the ground truth is retrieval relevance rather than
            # a generative answer. Do NOT fabricate missing answers (per engineering requirement).
            answer = ""

            metadata = {
                "original_id": raw_id,
                "title": q.get("title", ""),
                "has_ground_truth_answer": False,
                "task_type": "scientific_retrieval_evaluation",
                "relevant_corpus_ids": rel_cids,
                "num_relevant_docs": len(context_docs)
            }

            record = {
                "id": unique_id,
                "question": question,
                "context": context_docs,
                "answer": answer,
                "source": "scifact",
                "metadata": metadata
            }
            processed_records.append(record)

        logger.info(f"[SciFact] Successfully preprocessed {len(processed_records)} valid records.")
        return processed_records

    def save_jsonl(self, records: List[Dict[str, Any]], output_file: Path):
        """Writes records to a JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} records to {output_file} (File size: {round(output_file.stat().st_size / 1024, 2)} KB).")

    def prepare_evaluation_splits(self, all_records: List[Dict[str, Any]]):
        """
        Splits data reproducibly into:
        1. Training / experimentation (train.jsonl)
        2. Validation (validation.jsonl)
        3. Final testing (test_isolated.jsonl)
        4. Blind testing without answers (test_blind.jsonl)
        """
        sampling_cfg = self.config.get("sampling", {})
        train_ratio = sampling_cfg.get("train_ratio", 0.70)
        val_ratio = sampling_cfg.get("val_ratio", 0.15)
        test_ratio = sampling_cfg.get("test_ratio", 0.15)

        logger.info("=" * 60)
        logger.info("Preparing RSI-RAG Isolated Evaluation Splits")
        logger.info(f"Splits ratio: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%} (Seed: {self.seed})")

        # Group by source to ensure balanced representation in each split
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_records:
            src = r.get("source", "unknown")
            by_source.setdefault(src, []).append(r)

        train_records, val_records, test_records = [], [], []

        for src, items in by_source.items():
            shuffled = list(items)
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_records.extend(shuffled[:n_train])
            val_records.extend(shuffled[n_train:n_train + n_val])
            test_records.extend(shuffled[n_train + n_val:])

        # Shuffle each split again
        random.shuffle(train_records)
        random.shuffle(val_records)
        random.shuffle(test_records)

        # 1. Save Train split
        self.save_jsonl(train_records, self.eval_dir / "train.jsonl")

        # 2. Save Validation split
        self.save_jsonl(val_records, self.eval_dir / "validation.jsonl")

        # 3. Save Final Testing Set (ISOLATED - Expected answers preserved only in test_isolated)
        self.save_jsonl(test_records, self.eval_dir / "test_isolated.jsonl")

        # 4. Save Blind Testing Set for Recursive Self-Improvement Optimization Loops
        # Expected answers and sensitive ground-truth metadata are stripped so optimization loops cannot cheat!
        blind_test_records = []
        for r in test_records:
            blind_r = {
                "id": r["id"],
                "question": r["question"],
                "context": r["context"],
                "answer": "",  # Stripped for safe blind evaluation
                "source": r["source"],
                "metadata": {
                    "original_id": r.get("metadata", {}).get("original_id"),
                    "blind_eval": True
                }
            }
            blind_test_records.append(blind_r)

        self.save_jsonl(blind_test_records, self.eval_dir / "test_blind.jsonl")

        logger.info(f"Evaluation splits prepared:")
        logger.info(f"  - Train:         {len(train_records)} records -> {self.eval_dir / 'train.jsonl'}")
        logger.info(f"  - Validation:    {len(val_records)} records -> {self.eval_dir / 'validation.jsonl'}")
        logger.info(f"  - Test Isolated: {len(test_records)} records -> {self.eval_dir / 'test_isolated.jsonl'}")
        logger.info(f"  - Test Blind:    {len(blind_test_records)} records -> {self.eval_dir / 'test_blind.jsonl'}")

    def run(self, dataset_name: Optional[str] = None):
        """Runs preprocessing for specified or all datasets."""
        logger.info("=" * 60)
        logger.info("Starting RSI-RAG Preprocessing Pipeline")
        logger.info("=" * 60)

        targets = []
        if dataset_name and dataset_name.lower() != "all":
            targets = [dataset_name.lower()]
        else:
            for ds_key, ds_val in self.config.get("datasets", {}).items():
                if ds_val.get("enabled", True):
                    targets.append(ds_key)

        all_processed = []

        for target in targets:
            try:
                if target == "hotpotqa":
                    records = self.preprocess_hotpotqa()
                elif target == "ragbench":
                    records = self.preprocess_ragbench()
                elif target == "scifact":
                    records = self.preprocess_scifact()
                else:
                    logger.error(f"Unknown dataset: {target}")
                    continue

                # Save individual dataset JSONL
                out_path = self.processed_dir / f"{target}.jsonl"
                self.save_jsonl(records, out_path)
                all_processed.extend(records)

            except Exception as e:
                logger.error(f"Error preprocessing {target}: {e}", exc_info=True)
                raise

        # Save combined processed JSONL
        if all_processed:
            combined_path = self.processed_dir / "combined.jsonl"
            self.save_jsonl(all_processed, combined_path)

            # Prepare evaluation splits
            self.prepare_evaluation_splits(all_processed)

        logger.info("=" * 60)
        logger.info("Preprocessing Pipeline Completed Successfully.")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RSI-RAG Dataset Preprocessor")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        help="Dataset to preprocess (hotpotqa, ragbench, scifact, or all)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    preprocessor = DatasetPreprocessor(config_path=args.config)
    preprocessor.run(dataset_name=args.dataset)

if __name__ == "__main__":
    main()
