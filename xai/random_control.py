"""
random_control.py — Deterministic random-ranking negative control.

Produces a single uniformly random permutation of the 10 feature names,
seeded with 42 for full reproducibility.

This is NOT an explanation method. It is a negative control used in Step 6
(building reduced-input variants) and Step 7 (conditional §7.B retrainings)
to provide a fidelity floor: IG and KernelSHAP must outperform random
feature selection to demonstrate that their rankings are informative.

Usage:
    from xai.random_control import random_ranking
    ranking = random_ranking()  # list of (feature, rank_position) pairs
"""

import random
from xai.training_stats import PATH_SCALAR_FEATURES

RANDOM_SEED = 42


def random_ranking(seed=RANDOM_SEED):
    """
    Return a deterministic random permutation of the 10 path scalar features.

    Args:
        seed: random seed (default 42, locked per THESIS_DECISIONS)

    Returns:
        ranking: list of (feature_name, pseudo_score) sorted descending,
                 where pseudo_score is just (10 - rank_index) to mimic the
                 (feature, score) format used by IG and KernelSHAP rankings.
    """
    rng = random.Random(seed)
    features = list(PATH_SCALAR_FEATURES)
    rng.shuffle(features)
    # Assign descending pseudo-scores so the format is consistent
    ranking = [(feat, float(len(features) - i)) for i, feat in enumerate(features)]
    return ranking


def get_random_feature_order(seed=RANDOM_SEED):
    """
    Return just the ordered list of feature names (most 'important' first).
    Convenience wrapper around random_ranking().
    """
    return [feat for feat, _ in random_ranking(seed=seed)]


if __name__ == '__main__':
    r = random_ranking()
    print("Random ranking (seed=42):")
    for rank, (feat, score) in enumerate(r, 1):
        print(f"  {rank}. {feat}")
