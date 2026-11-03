"""
Selector engine: Tournament Selection, Elite Preservation, and Pareto Frontier calculation.
"""
from typing import List, Dict, Any, Tuple
import random
from backend.app.models import RAGStrategy

class StrategySelector:
    @staticmethod
    def tournament_selection(population: List[RAGStrategy], tournament_size: int = 3) -> RAGStrategy:
        """Selects the best individual among a randomly sampled tournament subset."""
        candidates = random.sample(population, min(tournament_size, len(population)))
        candidates.sort(key=lambda s: s.fitness if s.fitness is not None else -1.0, reverse=True)
        return candidates[0]

    @staticmethod
    def extract_elites(population: List[RAGStrategy], count: int) -> List[RAGStrategy]:
        """Returns the top-N fittest strategies without mutation."""
        sorted_pop = sorted(
            population,
            key=lambda s: s.fitness if s.fitness is not None else -1.0,
            reverse=True
        )
        return [s.model_copy(deep=True) for s in sorted_pop[:count]]

    @staticmethod
    def compute_pareto_frontier(population: List[RAGStrategy]) -> List[RAGStrategy]:
        """
        Computes the Pareto Frontier for multi-objective optimization:
        Objective 1: Maximize Fitness (composite accuracy/groundedness)
        Objective 2: Minimize Latency (latency_ms)

        A strategy A dominates B if:
        fitness_A >= fitness_B AND latency_A <= latency_B AND (at least one is strictly better).
        """
        valid_pop = [s for s in population if s.fitness is not None and s.latency_ms is not None]
        if not valid_pop:
            return population[:1]

        pareto_front = []
        for cand in valid_pop:
            is_dominated = False
            for other in valid_pop:
                if other.id == cand.id:
                    continue
                # other dominates cand if other has better or equal fitness and lower or equal latency
                if (other.fitness >= cand.fitness and other.latency_ms <= cand.latency_ms) and \
                   (other.fitness > cand.fitness or other.latency_ms < cand.latency_ms):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(cand)

        # Sort Pareto front by fitness descending
        pareto_front.sort(key=lambda s: s.fitness, reverse=True)
        return pareto_front

    @staticmethod
    def select_champion(population: List[RAGStrategy], prefer_low_latency: bool = False) -> RAGStrategy:
        """
        Selects champion pipeline.
        If prefer_low_latency is True, picks from the Pareto frontier with the best fitness-to-latency ratio.
        Otherwise picks the highest absolute fitness.
        """
        pareto = StrategySelector.compute_pareto_frontier(population)
        if not pareto:
            sorted_pop = sorted(population, key=lambda s: s.fitness or 0.0, reverse=True)
            return sorted_pop[0]

        if not prefer_low_latency:
            return pareto[0]

        # Multi-objective score: fitness / (latency_ms ** 0.3)
        best_strat = pareto[0]
        best_ratio = -1.0
        for s in pareto:
            ratio = (s.fitness or 0.0) / (max(10.0, s.latency_ms or 100.0) ** 0.25)
            if ratio > best_ratio:
                best_ratio = ratio
                best_strat = s
        return best_strat
