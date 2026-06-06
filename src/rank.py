import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ modules resolve correctly whether called from root or src/
sys.path.append(str(Path(__file__).parent))

from data_ingestion import load_candidates
from embeddings import EmbeddingModel, CandidateFAISS
from ranking import calculate_final_scores
from reasoning import generate_reasoning

# ─────────────────────────────────────────────────────────────────────────────
# FUNNEL CONSTANTS
#
# SEMANTIC_POOL:  How many candidates FAISS returns (Stage 1 filter).
#                 FAISS search is O(n) but written in C — extremely fast even
#                 for 2M candidates. This is the cheap stage.
#
# SCORE_POOL:     How many of those FAISS results we pass to the heavy Python
#                 scoring (jd_scorer + heuristics). This is the expensive stage.
#                 Kept at 5000 so the 5-minute budget is safe regardless of
#                 total dataset size. On a 2M candidate pool, FAISS still
#                 returns results in <1s; scoring 5000 takes ~2-3s.
#
# OUTPUT_ROWS:    Final ranked rows written to CSV. Spec requires exactly 100.
# ─────────────────────────────────────────────────────────────────────────────
SEMANTIC_POOL = 5_000   # candidates returned by FAISS (never hardcode dataset size)
SCORE_POOL    = 5_000   # candidates passed to heavy Python scoring
OUTPUT_ROWS   = 100     # rows in submission CSV


def main():
    parser = argparse.ArgumentParser(
        description="Online Ranking Script — 5-minute CPU budget"
    )
    parser.add_argument('--candidates',  required=True, help="Path to candidates .json/.jsonl/.jsonl.gz")
    parser.add_argument('--embeddings',  required=True, help="Path to precomputed embeddings .npy file")
    parser.add_argument('--jd',          required=True, help="Path to the Job Description text file")
    parser.add_argument('--out',         required=True, help="Output CSV path (e.g. team_xxx.csv)")
    parser.add_argument('--semantic-pool', type=int, default=SEMANTIC_POOL,
                        help=f"How many candidates FAISS retrieves (default: {SEMANTIC_POOL})")
    args = parser.parse_args()

    t_start = time.time()
    print("--- Starting Online Ranking Phase ---")

    # ── Step 1: Load data & pre-computed embeddings ───────────────────────────
    print("Loading data and embeddings...")
    candidates = load_candidates(args.candidates)   # dynamic — any dataset size
    embeddings = np.load(args.embeddings)

    n_candidates = len(candidates)
    if n_candidates != len(embeddings):
        raise ValueError(
            f"Mismatch: {n_candidates} candidates but {len(embeddings)} embeddings. "
            f"Re-run precompute.py on this dataset."
        )
    print(f"Dataset size: {n_candidates:,} candidates")

    # ── Step 2: Build FAISS index (fast C extension — handles millions of vectors) ─
    print("Building FAISS index...")
    faiss_index = CandidateFAISS(embedding_dim=384)
    faiss_index.add_embeddings(embeddings)

    # ── Step 3: Embed the Job Description ─────────────────────────────────────
    print("Embedding Job Description...")
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    embedder = EmbeddingModel('all-MiniLM-L6-v2')
    jd_embedding = embedder.generate_embeddings([jd_text])[0]

    # ── Step 4: FAISS Semantic Filter (Stage 1 — cheap, fast) ─────────────────
    # FAISS only returns the top-K semantically closest candidates.
    # This is the ONLY place we touch the full dataset.
    # k is capped at the actual dataset size to avoid FAISS errors on small inputs.
    k_semantic = min(args.semantic_pool, n_candidates)
    print(f"Semantic filter: FAISS retrieving top {k_semantic:,} of {n_candidates:,} candidates...")
    distances, indices = faiss_index.search(jd_embedding, k=k_semantic)

    # ── Step 5: Heavy Python Scoring (Stage 2 — only on semantic pool) ────────
    # jd_scorer + heuristics run on at most SCORE_POOL candidates, NOT the full DB.
    # This is the guarantee that keeps us under 5 minutes on any dataset size.
    pool = min(SCORE_POOL, k_semantic)
    print(f"Behavioral scoring: running on top {pool:,} semantic candidates...")
    final_rankings = calculate_final_scores(
        candidates,
        distances[:pool],
        indices[:pool]
    )

    # ── Step 6: Take top 100 and generate reasoning ────────────────────────────
    top_n = min(OUTPUT_ROWS, len(final_rankings))
    print(f"Generating reasoning for top {top_n} candidates...")
    output_rows = []

    for rank_idx, res in enumerate(final_rankings[:top_n]):
        cand = res['candidate']
        c_id = cand.get('candidate_id')
        score = res['final_score']

        reasoning = generate_reasoning(
            cand,
            rank=rank_idx + 1,
            behavioral_result=res['behavioral_result'],
            jd_result=res['jd_result']
        )

        output_rows.append({
            'candidate_id': c_id,
            'score':        round(score, 4),
            'reasoning':    reasoning
        })

    # ── Step 7: Final tie-break sort, then assign ranks ───────────────────────
    # Per submission spec §3: score DESC; on ties, candidate_id ASC.
    output_rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    for i, row in enumerate(output_rows):
        row['rank'] = i + 1

    # ── Step 8: Write CSV in exact required column order ──────────────────────
    df = pd.DataFrame(output_rows)[['candidate_id', 'rank', 'score', 'reasoning']]
    df.to_csv(args.out, index=False, encoding='utf-8')

    elapsed = time.time() - t_start
    print(f"\n[OK] Success! Output written to: {args.out}")
    print(f"[TIME]  Total online ranking time : {elapsed:.1f}s  (budget: 300s)")
    print(f"[INFO]  Candidates in dataset      : {n_candidates:,}")
    print(f"[INFO]  Semantic pool (FAISS)       : {k_semantic:,}")
    print(f"[INFO]  Scored (Python heuristics)  : {pool:,}")
    print(f"[INFO]  Rows in output              : {len(output_rows)}")


if __name__ == '__main__':
    main()
