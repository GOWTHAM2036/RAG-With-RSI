"""
Evolutionary Experiment Runner: executes generations of pipeline optimization.
"""
from typing import List, Dict, Any, Optional
import time
import json
import uuid
from pathlib import Path
from backend.app.models import (
    RAGStrategy,
    BenchmarkDataset,
    Document,
    ExperimentConfig,
    GenerationStats,
    OptimizationRunSummary
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

    def run_evolution(
        self,
        documents: List[Document],
        benchmark: BenchmarkDataset
    ) -> OptimizationRunSummary:
        start_time = time.time()
        pop_size = self.config.population_size
        num_gens = self.config.num_generations

        # 1. Initialize Generation 0 Population
        seeds = StrategyGenerator.heuristic_seed_strategies()
        population: List[RAGStrategy] = list(seeds[:min(len(seeds), pop_size)])
        while len(population) < pop_size:
            population.append(StrategyGenerator.random_strategy(generation=0))

        generation_history: List[GenerationStats] = []
        all_evaluated_strategies: List[RAGStrategy] = []

        weights = {
            "faithfulness": self.config.weight_faithfulness,
            "relevance": self.config.weight_relevance,
            "hit_rate": self.config.weight_hit_rate,
            "mrr": self.config.weight_mrr,
            "latency_penalty": self.config.weight_latency_penalty
        }

        for gen in range(num_gens):
            print(f"--- [EvoRAG] Starting Generation {gen + 1}/{num_gens} (Pop: {len(population)}) ---")

            # Evaluate each individual in current population
            for ind in population:
                ind.generation = gen
                report = BenchmarkRunner.run_strategy_benchmark(
                    strategy=ind,
                    documents=documents,
                    benchmark=benchmark,
                    weights=weights
                )
                all_evaluated_strategies.append(ind)

            # Sort population by fitness
            population.sort(key=lambda s: s.fitness if s.fitness is not None else -1.0, reverse=True)
            fitness_values = [s.fitness for s in population if s.fitness is not None]

            best_f = max(fitness_values) if fitness_values else 0.0
            avg_f = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
            worst_f = min(fitness_values) if fitness_values else 0.0
            champion = population[0].model_copy(deep=True)

            gen_stat = GenerationStats(
                generation=gen,
                best_fitness=round(best_f, 4),
                avg_fitness=round(avg_f, 4),
                worst_fitness=round(worst_f, 4),
                champion_strategy=champion,
                population=[s.model_copy(deep=True) for s in population]
            )
            generation_history.append(gen_stat)

            # If last generation, break
            if gen == num_gens - 1:
                break

            # 2. Formulate Next Generation
            elite_count = max(1, int(pop_size * self.config.elite_ratio))
            elites = StrategySelector.extract_elites(population, count=elite_count)
            next_population: List[RAGStrategy] = list(elites)

            while len(next_population) < pop_size:
                # Tournament select two parents
                p1 = StrategySelector.tournament_selection(population)
                p2 = StrategySelector.tournament_selection(population)

                # Crossover
                if random_should_occur(self.config.crossover_rate):
                    offspring = StrategyGenerator.crossover(p1, p2, generation=gen + 1)
                else:
                    offspring = p1.model_copy(deep=True)
                    offspring.id = f"strat-{str(uuid.uuid4())[:8]}"
                    offspring.generation = gen + 1

                # Mutation
                mutated_offspring = StrategyGenerator.mutate(offspring, mutation_rate=self.config.mutation_rate)
                mutated_offspring.generation = gen + 1
                next_population.append(mutated_offspring)

            population = next_population

        # Calculate final Pareto frontier over all evaluated strategies
        pareto_front = StrategySelector.compute_pareto_frontier(all_evaluated_strategies)
        overall_champion = StrategySelector.select_champion(all_evaluated_strategies)

        summary = OptimizationRunSummary(
            id=self.run_id,
            name=self.config.name,
            status="completed",
            generations_completed=num_gens,
            total_generations=num_gens,
            champion_strategy=overall_champion,
            generation_history=generation_history,
            pareto_frontier=pareto_front,
            start_time=start_time,
            end_time=time.time()
        )

        # Save to disk
        self._save_run_summary(summary)
        return summary

    def _save_run_summary(self, summary: OptimizationRunSummary):
        file_path = self.storage_dir / f"{summary.id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

def random_should_occur(probability: float) -> bool:
    import random
    return random.random() < probability
