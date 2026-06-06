import argparse
import sys
import numpy as np
from pathlib import Path

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from data_ingestion import load_candidates
from features import stringify_candidate
from embeddings import EmbeddingModel

def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for all candidates to save time during online ranking.")
    parser.add_argument('--candidates', required=True, help="Path to candidates json/jsonl")
    parser.add_argument('--out_embeddings', required=True, help="Path to save the .npy embeddings file")
    args = parser.parse_args()
    
    print("--- Starting Offline Pre-computation ---")
    candidates = load_candidates(args.candidates)
    
    print("Stringifying candidates...")
    strings = [stringify_candidate(c) for c in candidates]
    
    print("Initializing embedding model...")
    embedder = EmbeddingModel('all-MiniLM-L6-v2')
    
    print("Generating embeddings (this may take a while for large datasets)...")
    embeddings = embedder.generate_embeddings(strings)
    
    # Save embeddings to disk
    np.save(args.out_embeddings, embeddings)
    print(f"Successfully saved {len(embeddings)} embeddings to {args.out_embeddings}")

if __name__ == '__main__':
    main()
