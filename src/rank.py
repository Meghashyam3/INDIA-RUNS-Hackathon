import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_ingestion import load_candidates
from embeddings import EmbeddingModel, CandidateFAISS
from ranking import calculate_final_scores
from reasoning import generate_reasoning

# How many candidates pass through each stage of the funnel
SEMANTIC_POOL = 5_000   # FAISS returns this many closest candidates
OUTPUT_ROWS   = 100     # Final rows written to CSV


def main():
    parser = argparse.ArgumentParser(description="Online Ranking Script — runs in under 5 minutes")
    parser.add_argument('--candidates',    required=True, help="Path to candidates .json/.jsonl/.jsonl.gz")
    parser.add_argument('--embeddings',    required=True, help="Path to pre-computed embeddings .npy file")
    parser.add_argument('--jd',            required=True, help="Path to Job Description text file")
    parser.add_argument('--out',           required=True, help="Output CSV path (e.g. team_xxx.csv)")
    parser.add_argument('--semantic-pool', type=int, default=SEMANTIC_POOL,
                        help=f"How many candidates FAISS retrieves (default: {SEMANTIC_POOL})")
    args = parser.parse_args()

    t_start = time.time()
    print("--- Starting Online Ranking ---")

    # Step 1: Load candidates and pre-computed embeddings
    print("Loading data and embeddings...")
    candidates = load_candidates(args.candidates)
    embeddings = np.load(args.embeddings)

    if len(candidates) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(candidates)} candidates but {len(embeddings)} embeddings. "
            "Re-run precompute.py on this dataset."
        )
    print(f"Dataset size: {len(candidates):,} candidates")

    # Step 2: Build FAISS index
    print("Building FAISS index...")
    faiss_index = CandidateFAISS(embedding_dim=384)
    faiss_index.add_embeddings(embeddings)

    # Step 3: Embed the Job Description
    print("Embedding Job Description...")
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    embedder = EmbeddingModel('all-MiniLM-L6-v2')
    jd_embedding = embedder.generate_embeddings([jd_text])[0]

    # Step 4: FAISS semantic filter — returns top-K closest candidates
    k = min(args.semantic_pool, len(candidates))
    print(f"Semantic filter: retrieving top {k:,} of {len(candidates):,} candidates...")
    distances, indices = faiss_index.search(jd_embedding, k=k)

    # Step 5: Score the FAISS results with JD + behavioral scoring
    print(f"Scoring top {k:,} candidates...")
    final_rankings = calculate_final_scores(candidates, distances, indices)

    # Step 6: Take the top 100 and generate reasoning
    top_n = min(OUTPUT_ROWS, len(final_rankings))
    print(f"Generating reasoning for top {top_n} candidates...")
    output_rows = []

    for rank_idx, res in enumerate(final_rankings[:top_n]):
        cand = res['candidate']
        reasoning = generate_reasoning(
            cand,
            rank=rank_idx + 1,
            behavioral_result=res['behavioral_result'],
            jd_result=res['jd_result']
        )
        output_rows.append({
            'candidate_id': cand.get('candidate_id'),
            'score':        round(res['final_score'], 4),
            'reasoning':    reasoning
        })

    # Step 7: Final sort and rank assignment
    output_rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    for i, row in enumerate(output_rows):
        row['rank'] = i + 1

    # Step 8: Write CSV
    df = pd.DataFrame(output_rows)[['candidate_id', 'rank', 'score', 'reasoning']]
    df.to_csv(args.out, index=False, encoding='utf-8')

    elapsed = time.time() - t_start
    print(f"\n[OK] Written to: {args.out}")
    print(f"[TIME] {elapsed:.1f}s  (budget: 300s)")
    print(f"[INFO] Total candidates: {len(candidates):,}")
    print(f"[INFO] FAISS pool: {k:,}")
    print(f"[INFO] Output rows: {len(output_rows)}")


if __name__ == '__main__':
    main()
