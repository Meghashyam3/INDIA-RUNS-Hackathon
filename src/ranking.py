"""
Final Ranking Engine

Combines three scoring signals into one final score:
  final_score = (0.35 * semantic_score) + (0.45 * jd_score) + (0.20 * behavioral_score)

JD skill match is weighted highest because it captures intent, not just similarity.
"""
import numpy as np
from heuristics import compute_behavioral_multiplier
from jd_scorer import compute_jd_skill_score

W_SEMANTIC   = 0.35
W_JD         = 0.45
W_BEHAVIORAL = 0.20


def calculate_final_scores(candidates, distances, indices):
    """
    Scores each candidate from the FAISS results and returns them sorted best-first.
    """
    results = []

    for i, idx in enumerate(indices):
        candidate = candidates[idx]
        distance = distances[i]

        # Convert L2 distance to 0-1 similarity score
        semantic_score = 1.0 / (1.0 + float(distance))

        jd_result = compute_jd_skill_score(candidate)
        behavioral_result = compute_behavioral_multiplier(candidate)

        final_score = (
            (W_SEMANTIC   * semantic_score)
            + (W_JD       * jd_result['score'])
            + (W_BEHAVIORAL * behavioral_result['multiplier'])
        )

        results.append({
            'candidate':         candidate,
            'final_score':       round(final_score, 6),
            'semantic_score':    round(semantic_score, 6),
            'jd_skill_score':    round(jd_result['score'], 6),
            'multiplier':        behavioral_result['multiplier'],
            'behavioral_result': behavioral_result,
            'jd_result':         jd_result,
        })

    # Sort: highest score first; ties broken by candidate_id ascending (spec rule)
    results.sort(key=lambda x: (-x['final_score'], x['candidate'].get('candidate_id', '')))
    return results
