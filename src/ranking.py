"""
Final Ranking Engine

Combines three independent scoring signals into one final score:

  final_score = (W_SEM * semantic_score) 
              + (W_JD  * jd_skill_score)
              + (W_BEH * normalized_behavioral_score)

Weights are calibrated against the hackathon's NDCG@10 (0.50 weight) metric,
which means the top 10 must be near-perfect. JD skill matching is the most 
important signal for that.
"""
import numpy as np
from typing import List, Dict, Any

from heuristics import compute_behavioral_multiplier
from jd_scorer import compute_jd_skill_score

# --- Scoring Weights ---
W_SEM = 0.35   # Semantic (FAISS) similarity
W_JD  = 0.45   # Explicit JD skill match (most important)
W_BEH = 0.20   # Behavioral reliability multiplier


def calculate_final_scores(
    candidates: List[Dict[str, Any]],
    distances: np.ndarray,
    indices: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Scores each retrieved candidate using the 3-component formula and returns
    a sorted list (best first).
    """
    results = []

    for i, idx in enumerate(indices):
        candidate = candidates[idx]
        distance = distances[i]

        # 1. Semantic Score: convert L2 distance to 0-1 similarity
        #    MiniLM L2 distances typically range 0–2, so this maps cleanly
        semantic_score = 1.0 / (1.0 + float(distance))

        # 2. JD Skill Score (0.0 – 1.0)
        jd_result = compute_jd_skill_score(candidate)
        jd_skill_score = jd_result['score']

        # 3. Behavioral Multiplier (0.0 – 1.0)
        behavioral_result = compute_behavioral_multiplier(candidate)
        multiplier = behavioral_result['multiplier']

        # 4. Combine
        final_score = (
            (W_SEM * semantic_score)
            + (W_JD  * jd_skill_score)
            + (W_BEH * multiplier)
        )

        results.append({
            'candidate':          candidate,
            'final_score':        round(final_score, 6),
            'semantic_score':     round(semantic_score, 6),
            'jd_skill_score':     round(jd_skill_score, 6),
            'multiplier':         multiplier,
            'behavioral_result':  behavioral_result,
            'jd_result':          jd_result,
        })

    # Sort descending by final score; on ties sort by candidate_id ascending (spec rule)
    results.sort(key=lambda x: (-x['final_score'], x['candidate'].get('candidate_id', '')))
    return results
