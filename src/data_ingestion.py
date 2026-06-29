import json
import gzip
from pathlib import Path


def load_candidates(filepath):
    """
    Loads candidate data from a .json, .jsonl, or .jsonl.gz file.
    Returns a list of candidate dicts.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {filepath}")

    candidates = []

    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))

    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            candidates = data if isinstance(data, list) else [data]

    elif path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .jsonl, or .jsonl.gz")

    print(f"Loaded {len(candidates)} candidates from {path.name}")
    return candidates
