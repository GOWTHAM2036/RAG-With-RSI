"""
Unit tests for evolutionary optimizer:
- StrategyGenerator (random, crossover, mutation)
- StrategySelector (Tournament, Elites, 3D Pareto frontier across accuracy, latency, cost)
- EvolutionaryExperimentRunner (Deterministic seeding, Train/Val/Test isolation, Persistence)
"""
import pytest
from backend.app.models import RAGStrategy, Document, BenchmarkDataset, BenchmarkPair, ExperimentConfig
from backend.app.optimizer.strategy_generator import StrategyGenerator
from backend.app.optimizer.selector import StrategySelector
from backend.app.optimizer.experiment_runner import EvolutionaryExperimentRunner

def test_random_strategy_generation():
    strat = StrategyGenerator.random_strategy(generation=1)
    assert strat.chunk_size in StrategyGenerator.CHUNK_SIZES
    assert strat.chunk_overlap < strat.chunk_size
    assert 0.0 <= strat.hybrid_alpha <= 1.0

def test_crossover_and_mutation():
    p1 = StrategyGenerator.heuristic_seed_strategies()[0]
    p2 = StrategyGenerator.heuristic_seed_strategies()[1]
    child = StrategyGenerator.crossover(p1, p2, generation=1)
    assert child.generation == 1
    assert child.chunk_overlap < child.chunk_size

    mutated = StrategyGenerator.mutate(child, mutation_rate=1.0)
    assert mutated.chunk_overlap < mutated.chunk_size

def test_3d_pareto_dominance_with_cost():
    # s1: high accuracy, low latency, low cost
    s1 = RAGStrategy(name="Optimal-Candidate", measured_accuracy=0.95, measured_latency_ms=80.0, measured_cost_usd=0.0001)
    # s2: high accuracy, high latency, high cost
    s2 = RAGStrategy(name="Expensive-Slow-Candidate", measured_accuracy=0.90, measured_latency_ms=300.0, measured_cost_usd=0.0020)
    # s3: moderate accuracy, ultra low latency, lowest cost
    s3 = RAGStrategy(name="Budget-Fast-Candidate", measured_accuracy=0.85, measured_latency_ms=30.0, measured_cost_usd=0.00005)

    # s1 dominates s2 in all three objectives
    assert StrategySelector.dominates(s1, s2)
    # s1 does NOT dominate s3 because s3 has lower latency and lower cost
    assert not StrategySelector.dominates(s1, s3)
    assert not StrategySelector.dominates(s3, s1)

    population = [s1, s2, s3]
    pareto = StrategySelector.compute_pareto_frontier(population)
    names = [s.name for s in pareto]

    assert "Optimal-Candidate" in names
    assert "Budget-Fast-Candidate" in names
    assert "Expensive-Slow-Candidate" not in names  # Dominated by s1

def test_evolutionary_run_and_persistence(tmp_path):
    docs = [
        Document(
            title="Genetic Algorithms",
            content="Genetic algorithms are search heuristics inspired by natural evolution. They use mutation, selection, and crossover."
        )
    ]
    train_benchmark = BenchmarkDataset(
        name="Train Benchmark",
        pairs=[
            BenchmarkPair(
                query="What operators do genetic algorithms use?",
                ground_truth_contexts=["They use mutation, selection, and crossover."],
                ground_truth_answer="Mutation, selection, and crossover."
            )
        ]
    )
    val_benchmark = BenchmarkDataset(
        name="Validation Benchmark",
        pairs=[
            BenchmarkPair(
                query="How are genetic algorithms inspired?",
                ground_truth_contexts=["inspired by natural evolution."],
                ground_truth_answer="By natural evolution."
            )
        ]
    )
    test_benchmark = BenchmarkDataset(
        name="Isolated Protected Test Benchmark",
        pairs=[
            BenchmarkPair(
                query="What are genetic algorithms?",
                ground_truth_contexts=["Genetic algorithms are search heuristics."],
                ground_truth_answer="Search heuristics."
            )
        ]
    )
    config = ExperimentConfig(
        population_size=4,
        num_generations=2,
        mutation_rate=0.2,
        crossover_rate=0.7,
        random_seed=42
    )
    runner = EvolutionaryExperimentRunner(config=config, storage_dir=tmp_path)
    summary = runner.run_evolution(
        documents=docs,
        train_benchmark=train_benchmark,
        val_benchmark=val_benchmark,
        test_benchmark=test_benchmark
    )

    assert summary.status == "completed"
    assert len(summary.generation_history) == 2
    assert summary.random_seed == 42
    assert summary.dataset_fingerprint != ""
    assert summary.validation_champion is not None
    assert summary.test_evaluation is not None
    assert summary.test_evaluation.composite_fitness is not None

    # Verify persistent registry was created
    registry = EvolutionaryExperimentRunner.load_champions_registry(storage_dir=tmp_path)
    assert len(registry) >= 1
    assert registry[0]["champion_id"] == summary.champion_strategy.id
    assert registry[0]["random_seed"] == 42
