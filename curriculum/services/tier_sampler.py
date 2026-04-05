# tier_sampler.py
"""
Converts a scalar competence score into a probability distribution over curriculum tiers & samples a tier for the next training batch.

Uses a power-law mapping:  S[k] = C ^ difficulty[k]
then normalises to get probabilities P[k] = S[k] / sum_j S[j].

Low competence  -> most mass on easy tiers.
High competence -> mass shifts towards harder tiers.
"""

import random
from typing import Dict, List, Optional


# Default difficulty values for the 5 CLEVR tiers.
DEFAULT_DIFFICULTY = {1: 1.0, 2: 2.25, 3: 3.5, 4: 5, 5: 6.5}

class TierSampler:
    """
    Samples a curriculum tier according to a competence-driven distribution.

    Parameters:
        difficulty : dict[int, float] | None
            Maps tier id -> difficulty exponent.
            If None, uses DEFAULT_DIFFICULTY.
    """

    def __init__(self, difficulty: Optional[Dict[int, float]] = None):
        self.difficulty = difficulty if difficulty is not None else DEFAULT_DIFFICULTY

    def tier_scores(self, competence: float) -> Dict[int, float]:
        """
        Compute un-normalised power-law scores for every tier.
        
        Calculation: Score = (Competence)^(Difficulty Exponent)
        """
        return {k: competence ** d for k, d in self.difficulty.items()}

    def tier_probabilities(self, competence: float) -> Dict[int, float]:
        """
        Returns normalised P[k] (probabilities) for every tier..

        Probability = (Tier Score) / (Sum of all Tier Scores).
        """
        scores = self.tier_scores(competence)
        total = sum(scores.values())

        # Guard against division by zero 
        if total == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        return {k: s / total for k, s in scores.items()}

    def sample_tier(self, competence: float) -> int:
        """
        Draw a single tier from the competence-derived distribution.

        Returns:
            int: The sampled tier id.
        """
        probs = self.tier_probabilities(competence)
        tiers = list(probs.keys())
        weights = [probs[t] for t in tiers]

        # Weighted random selection: Tiers with higher probabilities are more likely to be selected for the next training batch.
        return random.choices(tiers, weights=weights, k=1)[0]

    def get_tier_ids(self) -> List[int]:
        """Return sorted list of tier ids."""
        return sorted(self.difficulty.keys())
