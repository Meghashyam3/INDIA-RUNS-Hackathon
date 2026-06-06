# Redrob AI — Intelligent Candidate Ranking System
### India Runs Hackathon · Data & AI Challenge Submission

> **Architecture**: Hybrid Semantic Search + Explicit JD Matching + Behavioral Heuristics  
> **Constraint compliance**: CPU-only · <16 GB RAM · Ranking step <5 minutes · Zero external API calls

---

## 📋 Quick Overview

This system ranks candidates for a given Job Description using **three independent scoring signals**:

| Signal | Weight | Description |
|---|---|---|
| **JD Skill Match** | 45% | Explicit keyword matching against must-have and nice-to-have skills extracted from the JD |
| **Semantic Similarity** | 35% | FAISS vector search using `all-MiniLM-L6-v2` embeddings |
| **Behavioral Reliability** | 20% | Multiplier derived from `redrob_signals` (response rate, interview completion, recency, GitHub) |

The system also detects **honeypot candidates** (profiles with statistically impossible data) and applies JD-specific **disqualifier penalties** (wrong-domain titles, consulting-only careers, etc.).

---

## 🗂️ Repository Structure

```
India_Runs_AI_Hackathon/
├── src/
│   ├── data_ingestion.py    # Dynamic loader for .json / .jsonl / .jsonl.gz
│   ├── features.py          # Converts candidate JSON → dense text string for embedding
│   ├── embeddings.py        # Wraps sentence-transformers + FAISS index
│   ├── jd_scorer.py         # Explicit JD skill matching + disqualifier detection
│   ├── heuristics.py        # Behavioral multiplier + honeypot detection
│   ├── ranking.py           # 3-component final scoring formula
│   ├── reasoning.py         # Factual, hallucination-free reasoning generator
│   ├── precompute.py        # OFFLINE: generate and save candidate embeddings
│   └── rank.py              # ONLINE: 5-minute ranking inference script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_faiss.ipynb
│   └── 04_advanced_ranking.ipynb
├── validate_submission.py   # Official organizer validator (copied from bundle)
├── submission_metadata.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- The hackathon data bundle (`candidates.jsonl`, `job_description.docx.txt`) downloaded and placed in a `data/` folder

### Install dependencies

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

---

## 🚀 Reproducing the Submission

The pipeline is split into two steps as required by the compute constraints:

### Step 1 — Offline Pre-computation (one-time, may take >5 min for 100K candidates)

Generates and saves dense embeddings for all candidates to disk. This step can be run once offline; the embedding file is reused for every ranking run.

```bash
python src/precompute.py \
  --candidates ./data/candidates.jsonl \
  --out_embeddings ./data/candidate_embeddings.npy
```

### Step 2 — Online Ranking (the timed 5-minute step)

Loads pre-computed embeddings, runs FAISS search, applies behavioral scoring, generates reasoning, and outputs the submission CSV.

```bash
python src/rank.py \
  --candidates ./data/candidates.jsonl \
  --embeddings ./data/candidate_embeddings.npy \
  --jd ./data/job_description.docx.txt \
  --out ./submission.csv
```

### Step 3 — Validate before submitting

```bash
python validate_submission.py submission.csv
```

---

## 🧠 Architecture Deep-Dive

### Phase A: Offline Pre-computation

```
candidates.jsonl (100K)
        │
        ▼
 data_ingestion.py     ← handles .json / .jsonl / .jsonl.gz dynamically
        │
        ▼
   features.py         ← stringify: title + summary + skills + career + education
        │
        ▼
  embeddings.py        ← all-MiniLM-L6-v2 (384-dim, CPU-optimised)
        │
        ▼
 candidate_embeddings.npy   ← saved to disk
```

### Phase B: Online Ranking (<5 minutes)

```
candidate_embeddings.npy + candidates.jsonl
        │
        ├──► FAISS IndexFlatL2   ← millisecond vector search
        │
JD text ──► embed ──► FAISS search → top-K candidates
        │
        ├──► jd_scorer.py        ← explicit skill overlap (45% weight)
        │      • 30+ must-have / nice-to-have keywords
        │      • role disqualifier penalties
        │      • consulting-career penalty
        │
        ├──► heuristics.py       ← behavioral multiplier (20% weight)
        │      • response rate, interview completion, recency
        │      • honeypot detection (3 checks)
        │
        ├──► ranking.py          ← combines all 3 signals → final_score
        │
        └──► reasoning.py        ← per-candidate factual reasoning string
                                   (zero hallucination: only uses profile data)
        │
        ▼
  submission.csv (top 100)
```

### Scoring Formula

```
final_score = (0.45 × jd_skill_score)
            + (0.35 × semantic_score)
            + (0.20 × behavioral_multiplier)
```

### Honeypot Detection

The system applies 3 checks to naturally filter impossible profiles:
1. **Expert skill + 0 months**: ≥10 skills marked `expert` with `duration_months=0`
2. **YoE impossibility**: Stated years of experience >15 years beyond earliest career date
3. **Perfect assessments**: ≥5 Redrob assessment scores all ≥98/100

---

## 🛡️ Key Design Decisions

**Why not a single FAISS search?**  
Raw semantic similarity is easily fooled by keyword stuffers. A "Marketing Manager" whose summary mentions "I used AI tools for content" will have high cosine similarity to an AI Engineer JD. The explicit JD skill matcher (45% weight) fixes this.

**Why split pre-computation from ranking?**  
Generating embeddings for 100K candidates takes ~3-4 minutes on a standard laptop CPU. The spec allows pre-computation to exceed 5 minutes; only the *ranking* step must fit within the budget. Splitting the pipeline means the online step (FAISS + scoring) runs in ~15 seconds.

**Why template-based reasoning instead of an LLM?**  
The spec explicitly disallows external LLM API calls during ranking. Our reasoning engine injects only verified facts from each candidate's JSON profile into structured templates, ensuring zero hallucination risk at Stage 4 review.

---

## 📊 Notebooks (Exploratory Work)

The `notebooks/` folder documents the iterative development process:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Data loading, schema inspection, Pandas flattening |
| `02_feature_engineering.ipynb` | Candidate stringifier design and testing |
| `03_baseline_faiss.ipynb` | Baseline semantic search — identifying the keyword trap |
| `04_advanced_ranking.ipynb` | Full 3-component pipeline with behavioral re-ranking |

---

## 🔧 Compute Environment

- **Platform**: Windows 11, Intel CPU
- **Python**: 3.14.2
- **RAM**: 16 GB
- **GPU**: Not used (CPU-only inference)
- **Network during ranking**: None (all models cached locally)

---

## 📄 License

This submission is original work created for the Redrob India Runs Hackathon 2026.
