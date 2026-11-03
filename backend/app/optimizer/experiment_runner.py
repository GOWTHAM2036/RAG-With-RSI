"""
Evolutionary Experiment Runner: executes generations of pipeline optimization.
Implements:
- Deterministic seeding for complete reproducibility
- Strict Train / Validation / Protected Test dataset isolation protocol
- Multi-objective Pareto frontier tracking (accuracy, latency, cost)
- Persistent champion versioning registry and experiment history
"""
from typing import List, Dict, Any, Optional
import time
import json
import uuid
import random
import numpy as np
from pathlib import Path
from backend.app.models import (
    RAGStrategy,
    BenchmarkDataset,
    Document,
    ExperimentConfig,
    GenerationStats,
    OptimizationRunSummary,
    EvaluationReport
)
from backend.app.optimizer.strategy_generator import StrategyGenerator
from backend.app.optimizer.selector import StrategySelector
from backend.app.evaluation.benchmarks import BenchmarkRunner
from backend.app.config import settings

class EvolutionaryExperimentRunner:
    def __init__(self, config: ExperimentConfig, storage_dir: Optional[Path] = None):
        self.config = config
        self.storage_dir = storage_dir or settings.EXPERIMENTS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"exp-{str(uuid.uuid4())[:8]}"

    def _set_deterministic_seed(self):
        seed = self.config.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def run_evolution(
        self,
        documents: Optional[List[Document]] = None,
        train_benchmark: Optional[BenchmarkDataset] = None,
        val_benchmark: Optional[BenchmarkDataset] = None,
        test_benchmark: Optional[BenchmarkDataset] = None,
        benchmark: Optional[BenchmarkDataset] = None
    ) -> OptimizationRunSummary:
        """
        Executes evolutionary search adhering to strict dataset isolation:
        1. Evolve across generations evaluated exclusively on Train benchmark.
        2. Validate Pareto candidates on Validation benchmark to select Champion.
        3. Evaluate final champion strictly ONCE on Protected Test benchmark.
        """
        start_time = time.time()
        self._set_deterministic_seed()

        pop_size = self.config.population_size
        num_gens = self.config.num_generations

        # 1. Resolve Dataset Isolation Splits
        train_bench = train_benchmark or benchmark
        train_docs = documents or []

        if not train_bench:
            train_bench, train_docs = BenchmarkRunner.load_rsi_split("train", sample_limit=self.config.eval_sample_limit)

        if not val_benchmark:
            val_bench, val_docs = BenchmarkRunner.load_rsi_split("validation", sample_limit=self.config.eval_sample_limit)
        else:
            val_bench, val_docs = val_benchmark, train_docs

        if not test_benchmark:
            test_bench, test_docs = BenchmarkRunner.load_rsi_split("test_isolated", sample_limit=self.config.eval_sample_limit)
        else:
            test_bench, test_docs = test_benchmark, train_docs

        dataset_fingerprint = train_bench.fingerprint or BenchmarkRunner.compute_dataset_fingerprint(train_bench)

        # 2. Initialize Generation 0 Population
        seeds = StrategyGenerator.heuristic_seed_strategies()
        population: List[RAGStrategy] = list(seeds[:min(len(seeds), pop_size)])
        while len(population) < pop_size:
            population.append(StrategyGenerator.random_strategy(generation=0))

        generation_history: List[GenerationStats] = []
        all_train_evaluated: List[RAGStrategy] = []

        weights = {
            "faithfulness": self.config.weight_faithfulness,
            "relevance": self.config.weight_relevance,
            "ndcg": self.config.weight_hit_rate,
            "context_precision": self.config.weight_mrr
        }

        # 3. Evolutionary Generations on Training Set
        for gen in range(num_gens):
            # Evaluate each individual in current population on Train data
            for ind in population:
                ind.generation = gen
                report = BenchmarkRunner.run_strategy_benchmark(
                    strategy=ind,
                    documents=train_docs,
                    benchmark=train_bench,
                    weights=weights,
                    generator_provider=self.config.generator_provider,
                    sample_limit=self.config.eval_sample_limit
                )
                all_train_evaluated.append(ind.model_copy(deep=True))

            # Sort population by measured accuracy
            population.sort(
                key=lambda s: s.measured_accuracy if s.measured_accuracy is not None else (s.fitness or -1.0),
                reverse=True
            )
            accuracy_values = [s.measured_accuracy for s in population if s.measured_accuracy is not None]

            best_acc = max(accuracy_values) if accuracy_values else 0.0
            avg_acc = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
            worst_acc = min(accuracy_values) if accuracy_values else 0.0
            gen_champion = population[0].model_copy(deep=True)

            gen_stat = GenerationStats(
                generation=gen,
                best_fitness=round(best_acc, 4),
                avg_fitness=round(avg_acc, 4),
                worst_fitness=round(worst_acc, 4),
                champion_strategy=gen_champion,
                population=[s.model_copy(deep=True) for s in population]
            )
            generation_history.append(gen_stat)

            if gen == num_gens - 1:
                break

            # Formulate Next Generation
            elite_count = max(1, int(pop_size * self.config.elite_ratio))
            elites = StrategySelector.extract_elites(population, count=elite_count)
            next_population: List[RAGStrategy] = list(elites)

            while len(next_population) < pop_size:
                p1 = StrategySelector.tournament_selection(population)
                p2 = StrategySelector.tournament_selection(population)

                if random.random() < self.config.crossover_rate:
                    offspring = StrategyGenerator.crossover(p1, p2, generation=gen + 1)
                else:
                    offspring = p1.model_copy(deep=True)
                    offspring.id = f"strat-{str(uuid.uuid4())[:8]}"
                    offspring.generation = gen + 1

                mutated_offspring = StrategyGenerator.mutate(offspring, mutation_rate=self.config.mutation_rate)
                mutated_offspring.generation = gen + 1
                next_population.append(mutated_offspring)

            population = next_population

        # 4. Multi-Objective Pareto Frontier on Training Data
        train_pareto_front = StrategySelector.compute_pareto_frontier(
            all_train_evaluated,
            objectives=self.config.pareto_objectives
        )

        # 5. Validation Selection Phase (Zero Train Bias)
        # Re-evaluate Pareto candidates on the separate Validation dataset
        candidates_to_validate = train_pareto_front[:min(5, len(train_pareto_front))]
        val_evaluated: List[RAGStrategy] = []
        for cand in candidates_to_validate:
            val_strat = cand.model_copy(deep=True)
            BenchmarkRunner.run_strategy_benchmark(
                strategy=val_strat,
                documents=val_docs if val_docs else train_docs,
                benchmark=val_bench,
                weights=weights,
                generator_provider=self.config.generator_provider,
                sample_limit=self.config.eval_sample_limit
            )
            val_evaluated.append(val_strat)

        validation_champion = StrategySelector.select_champion(val_evaluated, mode="balanced")

        # 6. Protected Test Set Evaluation (Evaluated strictly ONCE)
        test_strat = validation_champion.model_copy(deep=True)
        test_report = BenchmarkRunner.run_strategy_benchmark(
            strategy=test_strat,
            documents=test_docs if test_docs else train_docs,
            benchmark=test_bench,
            weights=weights,
            generator_provider=self.config.generator_provider,
            sample_limit=self.config.eval_sample_limit
        )

        summary = OptimizationRunSummary(
            id=self.run_id,
            name=self.config.name,
            status="completed",
            generations_completed=num_gens,
            total_generations=num_gens,
            random_seed=self.config.random_seed,
            dataset_fingerprint=dataset_fingerprint,
            champion_strategy=validation_champion,
            validation_champion=validation_champion,
            test_evaluation=test_report,
            generation_history=generation_history,
            pareto_frontier=train_pareto_front,
            start_time=start_time,
            end_time=time.time()
        )

        # 7. Persistent Storage & Champion Registry
        self._save_run_summary(summary)
        self._register_champion(summary, test_report)

        return summary

    def _save_run_summary(self, summary: OptimizationRunSummary):
        file_path = self.storage_dir / f"{summary.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

    def _register_champion(self, summary: OptimizationRunSummary, test_report: EvaluationReport):
        registry_file = self.storage_dir / "champions_registry.json"
        registry: List[Dict[str, Any]] = []

        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except Exception:
                registry = []

        entry = {
            "version": len(registry) + 1,
            "champion_id": summary.champion_strategy.id,
            "champion_name": summary.champion_strategy.name,
            "run_id": summary.id,
            "timestamp": time.time(),
            "random_seed": summary.random_seed,
            "dataset_fingerprint": summary.dataset_fingerprint,
            "strategy": summary.champion_strategy.model_dump(),
            "test_accuracy": test_report.composite_fitness,
            "test_faithfulness": test_report.answer_metrics.faithfulness,
            "test_latency_ms": test_report.avg_latency_ms,
            "test_cost_usd": test_report.measured_cost_usd,
            "total_tokens": test_report.total_tokens
        }
        registry.append(entry)

        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @classmethod
    def load_champions_registry(cls, storage_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        target_dir = storage_dir or settings.EXPERIMENTS_DIR
        registry_file = target_dir / "champions_registry.json"
        if not registry_file.exists():
            return []
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
