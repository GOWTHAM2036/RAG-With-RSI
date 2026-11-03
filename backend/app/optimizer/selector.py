"""
Selector engine: Multi-Objective Pareto Frontier calculation, Tournament Selection, and Elitism.
Supports genuine 3D multi-objective optimization across:
- Accuracy (Maximized: Groundedness + Retrieval NDCG/MRR + Relevance)
- Latency (Minimized: End-to-End latency in ms)
- Cost (Minimized: Real measured token consumption and API/compute cost in USD)
"""
from typing import List, Dict, Any, Tuple, Optional
import random
from backend.app.models import RAGStrategy

class StrategySelector:
    OBJECTIVE_DIRECTIONS = {
        "accuracy": "max",
        "latency": "min",
        "cost": "min"
    }

    @staticmethod
    def _extract_objectives(strategy: RAGStrategy) -> Dict[str, float]:
        """Extracts the three core measured Pareto objective values from a strategy."""
        acc = strategy.measured_accuracy if strategy.measured_accuracy is not None else (strategy.fitness or 0.0)
        lat = strategy.measured_latency_ms if strategy.measured_latency_ms is not None else (strategy.latency_ms or 500.0)
        cost = strategy.measured_cost_usd if strategy.measured_cost_usd is not None else 0.0001
        return {
            "accuracy": float(acc),
            "latency": float(lat),
            "cost": float(cost)
        }

    @staticmethod
    def dominates(strat_a: RAGStrategy, strat_b: RAGStrategy, objectives: Optional[List[str]] = None) -> bool:
        """
        True Multi-Objective Pareto Dominance test.
        strat_a dominates strat_b if:
        1. For ALL objectives, strat_a is at least as good as strat_b.
        2. For AT LEAST ONE objective, strat_a is strictly better than strat_b.
        """
        objs = objectives or ["accuracy", "latency", "cost"]
        val_a = StrategySelector._extract_objectives(strat_a)
        val_b = StrategySelector._extract_objectives(strat_b)

        at_least_as_good_all = True
        strictly_better_one = False

        for obj in objs:
            dir_ = StrategySelector.OBJECTIVE_DIRECTIONS.get(obj, "max")
            a_val = val_a[obj]
            b_val = val_b[obj]

            if dir_ == "max":
                if a_val < b_val:
                    at_least_as_good_all = False
                    break
                if a_val > b_val:
                    strictly_better_one = True
            else:  # min
                if a_val > b_val:
                    at_least_as_good_all = False
                    break
                if a_val < b_val:
                    strictly_better_one = True

        return at_least_as_good_all and strictly_better_one

    @staticmethod
    def compute_pareto_frontier(
        population: List[RAGStrategy],
        objectives: Optional[List[str]] = None
    ) -> List[RAGStrategy]:
        """
        Computes the Non-Dominated Pareto Frontier across the specified objectives
        (default: accuracy [max], latency [min], cost [min]).
        """
        valid_pop = [s for s in population if (s.fitness is not None or s.measured_accuracy is not None)]
        if not valid_pop:
            return population[:1]

        pareto_front: List[RAGStrategy] = []
        for cand in valid_pop:
            is_dominated = False
            for other in valid_pop:
                if other.id == cand.id:
                    continue
                if StrategySelector.dominates(other, cand, objectives=objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                # Avoid duplicate identical configurations on the frontier
                if not any(p.id == cand.id for p in pareto_front):
                    pareto_front.append(cand)

        # Sort Pareto front by accuracy descending
        pareto_front.sort(
            key=lambda s: s.measured_accuracy if s.measured_accuracy is not None else (s.fitness or 0.0),
            reverse=True
        )
        return pareto_front

    @staticmethod
    def select_champion(
        population: List[RAGStrategy],
        mode: str = "balanced"  # balanced, max_accuracy, min_latency, min_cost
    ) -> RAGStrategy:
        """
        Selects a champion strategy from the non-dominated Pareto frontier based on selection mode:
        - 'balanced': Maximizes normalized trade-off: accuracy / ((latency_ms / 50)**0.2 * (cost / 1e-4)**0.1)
        - 'max_accuracy': Pareto individual with the highest accuracy
        - 'min_latency': Pareto individual with the lowest latency
        - 'min_cost': Pareto individual with the lowest measured cost
        """
        pareto = StrategySelector.compute_pareto_frontier(population)
        if not pareto:
            sorted_pop = sorted(population, key=lambda s: s.fitness or 0.0, reverse=True)
            return sorted_pop[0]

        if mode == "max_accuracy":
            return max(
                pareto,
                key=lambda s: s.measured_accuracy if s.measured_accuracy is not None else (s.fitness or 0.0)
            )

        if mode == "min_latency":
            return min(
                pareto,
                key=lambda s: s.measured_latency_ms if s.measured_latency_ms is not None else (s.latency_ms or 9999.0)
            )

        if mode == "min_cost":
            return min(
                pareto,
                key=lambda s: s.measured_cost_usd if s.measured_cost_usd is not None else 9999.0
            )

        # Balanced trade-off score
        best_strat = pareto[0]
        best_score = -1e9

        for s in pareto:
            objs = StrategySelector._extract_objectives(s)
            acc = max(0.01, objs["accuracy"])
            lat = max(10.0, objs["latency"])
            cost = max(1e-6, objs["cost"])

            # Harmonic multi-objective trade-off
            # Rewards high accuracy while dampening latency and token expense
            score = acc / ((lat / 50.0) ** 0.25 * (cost / 0.0001) ** 0.15)
            if score > best_score:
                best_score = score
                best_strat = s

        return best_strat

    @staticmethod
    def tournament_selection(population: List[RAGStrategy], tournament_size: int = 3) -> RAGStrategy:
        """Selects the best individual among a randomly sampled tournament subset."""
        candidates = random.sample(population, min(tournament_size, len(population)))
        candidates.sort(
            key=lambda s: s.fitness if s.fitness is not None else (s.measured_accuracy or -1.0),
            reverse=True
        )
        return candidates[0]

    @staticmethod
    def extract_elites(population: List[RAGStrategy], count: int) -> List[RAGStrategy]:
        """Returns the top-N fittest strategies without mutation."""
        sorted_pop = sorted(
            population,
            key=lambda s: s.fitness if s.fitness is not None else (s.measured_accuracy or -1.0),
            reverse=True
        )
        return [s.model_copy(deep=True) for s in sorted_pop[:count]]
