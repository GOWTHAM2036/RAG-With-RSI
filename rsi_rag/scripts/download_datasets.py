"""
Automated Dataset Downloader for RSI-RAG.
Downloads HotpotQA, RAGBench, and SciFact with configurable sampling limits,
network retry logic, verification, and local disk caching via save_to_disk().
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import load_config, setup_logger, retry_with_backoff, get_directory_size_mb

logger = setup_logger("downloader", log_file="logs/download.log")

class DatasetDownloader:
    def __init__(self, config_path: str = "config/dataset_config.yaml", force: bool = False):
        self.config = load_config(config_path)
        self.force = force
        self.storage_cfg = self.config.get("storage", {})
        self.raw_dir = Path(self.storage_cfg.get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.net_cfg = self.config.get("network", {})
        self.max_retries = self.net_cfg.get("max_retries", 3)
        self.retry_delay = self.net_cfg.get("retry_delay", 2.0)

    def is_cached(self, dataset_name: str) -> bool:
        """Checks if dataset has already been downloaded and saved to disk."""
        dest = self.raw_dir / dataset_name
        if not dest.exists():
            return False
        # Check if arrow or metadata files exist inside
        files = list(dest.rglob("*.arrow")) + list(dest.rglob("dataset_info.json"))
        return len(files) > 0

    def download_hotpotqa(self, sample_limit: Optional[int] = None) -> Path:
        """
        Downloads HotpotQA (distractor configuration).
        Splits: train and validation.
        """
        ds_cfg = self.config["datasets"]["hotpotqa"]
        limit = sample_limit or ds_cfg.get("sample_limit", 500)
        hf_name = ds_cfg.get("name", "hotpotqa/hotpot_qa")
        config_name = ds_cfg.get("config", "distractor")
        dest_path = self.raw_dir / "hotpotqa"

        if self.is_cached("hotpotqa") and not self.force:
            logger.info(f"[HotpotQA] Raw dataset already cached at {dest_path}. Use --force to re-download.")
            return dest_path

        logger.info(f"[HotpotQA] Downloading '{hf_name}' (config: '{config_name}', limit: {limit} samples per split)...")

        splits_to_fetch = ds_cfg.get("splits", ["train", "validation"])
        dataset_dict = {}

        for split in splits_to_fetch:
            logger.info(f"[HotpotQA] Streaming split '{split}'...")
            
            @retry_with_backoff(max_retries=self.max_retries, initial_delay=self.retry_delay)
            def fetch_split():
                stream = load_dataset(hf_name, config_name, split=split, streaming=True)
                samples = []
                pbar = tqdm(total=limit, desc=f"HotpotQA [{split}]", unit="sample")
                for item in stream.take(limit):
                    samples.append(item)
                    pbar.update(1)
                pbar.close()
                return Dataset.from_list(samples)

            dataset_dict[split] = fetch_split()
            logger.info(f"[HotpotQA] Collected {len(dataset_dict[split])} records for split '{split}'.")

        final_ds = DatasetDict(dataset_dict)
        dest_path.mkdir(parents=True, exist_ok=True)
        final_ds.save_to_disk(dest_path)

        size_mb = get_directory_size_mb(dest_path)
        logger.info(f"[HotpotQA] Successfully saved raw dataset to {dest_path} (Size: {size_mb} MB).")
        return dest_path

    def download_ragbench(self, sample_limit: Optional[int] = None) -> Path:
        """
        Downloads RAGBench.
        RAGBench requires specifying a task/domain config (e.g., 'hotpotqa', 'covidqa', etc.).
        Splits: train, validation, test.
        """
        ds_cfg = self.config["datasets"]["ragbench"]
        limit = sample_limit or ds_cfg.get("sample_limit", 200)
        hf_name = ds_cfg.get("name", "rungalileo/ragbench")
        config_name = ds_cfg.get("config", "hotpotqa")
        dest_path = self.raw_dir / "ragbench"

        if self.is_cached("ragbench") and not self.force:
            logger.info(f"[RAGBench] Raw dataset already cached at {dest_path}. Use --force to re-download.")
            return dest_path

        logger.info(f"[RAGBench] Downloading '{hf_name}' (config: '{config_name}', limit: {limit} samples per split)...")

        splits_to_fetch = ds_cfg.get("splits", ["train", "validation", "test"])
        dataset_dict = {}

        for split in splits_to_fetch:
            logger.info(f"[RAGBench] Streaming split '{split}'...")

            @retry_with_backoff(max_retries=self.max_retries, initial_delay=self.retry_delay)
            def fetch_split():
                stream = load_dataset(hf_name, config_name, split=split, streaming=True)
                samples = []
                pbar = tqdm(total=limit, desc=f"RAGBench [{split}]", unit="sample")
                for item in stream.take(limit):
                    samples.append(item)
                    pbar.update(1)
                pbar.close()
                return Dataset.from_list(samples)

            try:
                dataset_dict[split] = fetch_split()
                logger.info(f"[RAGBench] Collected {len(dataset_dict[split])} records for split '{split}'.")
            except Exception as e:
                logger.warning(f"[RAGBench] Warning loading split '{split}': {e}. Continuing with available splits.")

        if not dataset_dict:
            raise RuntimeError(f"Failed to fetch any splits for {hf_name} with config {config_name}")

        final_ds = DatasetDict(dataset_dict)
        dest_path.mkdir(parents=True, exist_ok=True)
        final_ds.save_to_disk(dest_path)

        size_mb = get_directory_size_mb(dest_path)
        logger.info(f"[RAGBench] Successfully saved raw dataset to {dest_path} (Size: {size_mb} MB).")
        return dest_path

    def download_scifact(self, sample_limit: Optional[int] = None) -> Path:
        """
        Downloads BeIR/scifact (corpus, queries, and qrels).
        """
        ds_cfg = self.config["datasets"]["scifact"]
        limit = sample_limit or ds_cfg.get("sample_limit", 200)
        hf_name = ds_cfg.get("name", "BeIR/scifact")
        qrels_name = ds_cfg.get("qrels_name", "BeIR/scifact-qrels")
        dest_path = self.raw_dir / "scifact"

        if self.is_cached("scifact") and not self.force:
            logger.info(f"[SciFact] Raw dataset already cached at {dest_path}. Use --force to re-download.")
            return dest_path

        logger.info(f"[SciFact] Downloading '{hf_name}' (corpus + queries + qrels, limit: {limit} queries)...")

        dataset_dict = {}

        # 1. Fetch Queries
        @retry_with_backoff(max_retries=self.max_retries, initial_delay=self.retry_delay)
        def fetch_queries():
            stream = load_dataset(hf_name, "queries", split="queries", streaming=True)
            samples = []
            pbar = tqdm(total=limit, desc="SciFact [queries]", unit="query")
            for item in stream.take(limit):
                samples.append(item)
                pbar.update(1)
            pbar.close()
            return Dataset.from_list(samples)

        dataset_dict["queries"] = fetch_queries()
        logger.info(f"[SciFact] Collected {len(dataset_dict['queries'])} queries.")

        # 2. Fetch Qrels
        @retry_with_backoff(max_retries=self.max_retries, initial_delay=self.retry_delay)
        def fetch_qrels():
            # Load test and train qrels
            qrels_ds = load_dataset(qrels_name)
            qrels_records = []
            for split_name in qrels_ds.keys():
                for item in qrels_ds[split_name]:
                    qrels_records.append(item)
            return Dataset.from_list(qrels_records)

        dataset_dict["qrels"] = fetch_qrels()
        logger.info(f"[SciFact] Collected {len(dataset_dict['qrels'])} qrels mappings.")

        # Extract corpus IDs needed for these queries to avoid downloading all 5,000+ corpus docs
        query_ids = set(str(q["_id"]) for q in dataset_dict["queries"])
        needed_corpus_ids = set()
        for qrel in dataset_dict["qrels"]:
            if str(qrel["query-id"]) in query_ids:
                needed_corpus_ids.add(str(qrel["corpus-id"]))

        logger.info(f"[SciFact] Queries map to {len(needed_corpus_ids)} specific ground-truth corpus documents.")

        # 3. Fetch Corpus (all needed documents + up to limit distractors)
        @retry_with_backoff(max_retries=self.max_retries, initial_delay=self.retry_delay)
        def fetch_corpus():
            stream = load_dataset(hf_name, "corpus", split="corpus", streaming=True)
            samples = []
            distractor_count = 0
            max_distractors = limit
            pbar = tqdm(desc="SciFact [corpus]", unit="doc")

            for item in stream:
                cid = str(item["_id"])
                if cid in needed_corpus_ids:
                    samples.append(item)
                    pbar.update(1)
                elif distractor_count < max_distractors:
                    samples.append(item)
                    distractor_count += 1
                    pbar.update(1)

                if len(needed_corpus_ids) > 0 and all(any(str(s["_id"]) == ncid for s in samples) for ncid in needed_corpus_ids) and distractor_count >= max_distractors:
                    break
            pbar.close()
            return Dataset.from_list(samples)

        dataset_dict["corpus"] = fetch_corpus()
        logger.info(f"[SciFact] Collected {len(dataset_dict['corpus'])} corpus documents (including negative distractors).")

        final_ds = DatasetDict(dataset_dict)
        dest_path.mkdir(parents=True, exist_ok=True)
        final_ds.save_to_disk(dest_path)

        size_mb = get_directory_size_mb(dest_path)
        logger.info(f"[SciFact] Successfully saved raw dataset to {dest_path} (Size: {size_mb} MB).")
        return dest_path

    def run(self, dataset_name: Optional[str] = None, sample_limit: Optional[int] = None):
        """Runs download for one or all enabled datasets."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Starting RSI-RAG Automated Dataset Downloader")
        logger.info("=" * 60)

        targets = []
        if dataset_name and dataset_name.lower() != "all":
            targets = [dataset_name.lower()]
        else:
            for ds_key, ds_val in self.config.get("datasets", {}).items():
                if ds_val.get("enabled", True):
                    targets.append(ds_key)

        logger.info(f"Target datasets to process: {targets}")
        results = {}

        for target in targets:
            try:
                if target == "hotpotqa":
                    results[target] = self.download_hotpotqa(sample_limit)
                elif target == "ragbench":
                    results[target] = self.download_ragbench(sample_limit)
                elif target == "scifact":
                    results[target] = self.download_scifact(sample_limit)
                else:
                    logger.error(f"Unknown dataset requested: '{target}'. Available: hotpotqa, ragbench, scifact.")
            except Exception as e:
                logger.error(f"Failed to download {target}: {e}", exc_info=True)
                raise

        elapsed = round(time.time() - t0, 2)
        logger.info("=" * 60)
        logger.info(f"Download pipeline completed in {elapsed}s.")
        for name, path in results.items():
            size = get_directory_size_mb(path)
            logger.info(f"  - {name}: {path} ({size} MB)")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RSI-RAG Dataset Downloader")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        help="Specific dataset to download (hotpotqa, ragbench, scifact, or all)"
    )
    parser.add_argument(
        "--sample-limit", "-n",
        type=int,
        default=None,
        help="Override sample limit for downloading"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset is already saved to disk"
    )
    args = parser.parse_args()

    downloader = DatasetDownloader(config_path=args.config, force=args.force)
    downloader.run(dataset_name=args.dataset, sample_limit=args.sample_limit)

if __name__ == "__main__":
    main()
