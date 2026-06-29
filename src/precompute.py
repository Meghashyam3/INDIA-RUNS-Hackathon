import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_ingestion import load_candidates
from features import stringify_candidate
from embeddings import EmbeddingModel


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for all candidates.")
    parser.add_argument('--candidates', required=True, help="Path to candidates .json/.jsonl/.jsonl.gz")
    parser.add_argument('--out_embeddings', required=True, help="Output path for the .npy embeddings file")
    args = parser.parse_args()

    print("--- Starting Offline Pre-computation ---")
    candidates = load_candidates(args.candidates)

    print("Converting candidates to text strings...")
    strings = [stringify_candidate(c) for c in candidates]

    print("Loading embedding model...")
    embedder = EmbeddingModel('all-MiniLM-L6-v2')

    print("Generating embeddings (this takes ~40 min for 100K candidates on CPU)...")
    embeddings = embedder.generate_embeddings(strings)

    out_path = Path(args.out_embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)
    print(f"Saved {len(embeddings)} embeddings to {args.out_embeddings}")


if __name__ == '__main__':
    main()
