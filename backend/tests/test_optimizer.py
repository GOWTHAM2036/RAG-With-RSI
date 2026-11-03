"""
Unit tests for evolutionary optimizer (StrategyGenerator, Selector, ExperimentRunner).
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

def test_pareto_frontier_selection():
    s1 = RAGStrategy(name="Fast-Accurate", fitness=0.9, latency_ms=100.0)
    s2 = RAGStrategy(name="Slow-Accurate", fitness=0.9, latency_ms=300.0)
    s3 = RAGStrategy(name="Fast-Inaccurate", fitness=0.6, latency_ms=50.0)
    s4 = RAGStrategy(name="Slow-Inaccurate", fitness=0.5, latency_ms=400.0)

    population = [s1, s2, s3, s4]
    pareto = StrategySelector.compute_pareto_frontier(population)

    names = [s.name for s in pareto]
    assert "Fast-Accurate" in names
    assert "Slow-Inaccurate" not in names  # Dominated by s1

def test_evolutionary_run(tmp_path):
    docs = [
        Document(
            title="Genetic Algorithms",
            content="Genetic algorithms are search heuristics inspired by natural evolution. They use mutation, selection, and crossover."
        )
    ]
    benchmark = BenchmarkDataset(
        name="Mini Benchmark",
        pairs=[
            BenchmarkPair(
                query="What operators do genetic algorithms use?",
                ground_truth_contexts=["They use mutation, selection, and crossover."],
                ground_truth_answer="Mutation, selection, and crossover."
            )
        ]
    )
    config = ExperimentConfig(
        population_size=4,
        num_generations=2,
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    runner = EvolutionaryExperimentRunner(config=config, storage_dir=tmp_path)
    summary = runner.run_evolution(documents=docs, benchmark=benchmark)

    assert summary.status == "completed"
    assert len(summary.generation_history) == 2
    assert summary.champion_strategy.fitness is not None
