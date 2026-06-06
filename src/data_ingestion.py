import json
import gzip
from pathlib import Path
from typing import List, Dict, Any

def load_candidates(filepath: str) -> List[Dict[str, Any]]:
    """
    Dynamically loads candidate data from either a standard .json file (array of objects),
    a .jsonl file (JSON lines), or a .jsonl.gz file (compressed JSON lines).
    
    Args:
        filepath (str): Path to the candidate data file.
        
    Returns:
        List[Dict]: A list of candidate profiles as Python dictionaries.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found at: {filepath}")

    candidates = []

    # Handle compressed .jsonl.gz files
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
                    
    # Handle standard .json array files
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If the json is an array of objects
            if isinstance(data, list):
                candidates = data
            else:
                candidates = [data]
                
    # Handle .jsonl text files
    elif path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .json, .jsonl, or .jsonl.gz")

    print(f"Successfully loaded {len(candidates)} candidates from {path.name}")
    return candidates

if __name__ == "__main__":
    # Test the loader with the sample dataset
    sample_path = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"
    try:
        data = load_candidates(sample_path)
        if data:
            print(f"Sample Candidate ID: {data[0].get('candidate_id')}")
    except Exception as e:
        print(f"Error loading sample: {e}")
