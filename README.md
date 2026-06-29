# Redrob AI — Intelligent Candidate Ranking System
### India Runs Hackathon · Data & AI Challenge Submission

> **Architecture**: Hybrid Semantic Search + Explicit JD Matching + Behavioral Heuristics  
> **Constraint compliance**: CPU-only · <16 GB RAM · Ranking step <5 minutes · Zero external API calls

---

## 📋 Quick Overview

This system ranks candidates for a given Job Description using **three independent scoring signals**:

| Signal | Weight | Description |
|---|---|---|
| **JD Skill Match** | 45% | Explicit keyword matching against must-have and nice-to-have skills from the JD |
| **Semantic Similarity** | 35% | FAISS vector search using `all-MiniLM-L6-v2` embeddings |
| **Behavioral Reliability** | 20% | Multiplier derived from `redrob_signals` (response rate, interview completion, recency, GitHub) |

The system also detects **honeypot candidates** (profiles with impossible data) and applies **role-based disqualifier penalties** (wrong-domain titles, full consulting-only careers, etc.).

---

## 🗂️ Repository Structure

```
CODE/
├── src/
│   ├── data_ingestion.py    # Loads .json / .jsonl / .jsonl.gz candidate files
│   ├── features.py          # Converts candidate JSON → text string for embedding
│   ├── embeddings.py        # SentenceTransformer wrapper + FAISS index
│   ├── jd_scorer.py         # JD skill matching + disqualifier detection
│   ├── heuristics.py        # Behavioral multiplier + honeypot detection
│   ├── ranking.py           # Combines all 3 signals into a final score
│   ├── reasoning.py         # Factual per-candidate reasoning generator
│   ├── precompute.py        # OFFLINE: generates and saves candidate embeddings
│   └── rank.py              # ONLINE: full ranking pipeline (runs in <5 minutes)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_faiss.ipynb
│   └── 04_advanced_ranking.ipynb
├── app.py                   # Streamlit sandbox demo
├── validate_submission.py   # Official organizer validator
├── submission_metadata.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- The hackathon data bundle (`candidates.jsonl`, `job_description.docx.txt`)

### Install dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Reproducing the Submission

The pipeline is split into two steps to meet the compute time constraint:

### Step 1 — Offline Pre-computation (run once, can take >5 min for 100K candidates)

Generates dense embeddings for every candidate and saves them to disk. Only needs to be run once — the file is reused for every ranking run.

```bash
python src/precompute.py \
  --candidates path/to/candidates.jsonl \
  --out_embeddings data/candidate_embeddings.npy
```

### Step 2 — Online Ranking (the timed 5-minute step)

Loads pre-computed embeddings, runs FAISS search, scores candidates, generates reasoning, and writes the submission CSV. Completes in ~30-60 seconds.

```bash
python src/rank.py \
  --candidates path/to/candidates.jsonl \
  --embeddings data/candidate_embeddings.npy \
  --jd path/to/job_description.docx.txt \
  --out team_VertexLabs.csv
```

### Step 3 — Validate before submitting

```bash
python validate_submission.py team_VertexLabs.csv
```

Expected output: `Submission is valid.`

### Step 4 — (Optional) Run the Streamlit Sandbox

```bash
streamlit run app.py
```

---

## 🧠 How It Works

### Phase A: Offline Pre-computation

```
candidates.jsonl
      │
      ▼
data_ingestion.py   ← handles .json / .jsonl / .jsonl.gz
      │
      ▼
features.py         ← builds text string: title + summary + skills + career + education
      │
      ▼
embeddings.py       ← all-MiniLM-L6-v2 (384-dim, CPU-optimised)
      │
      ▼
candidate_embeddings.npy
```

### Phase B: Online Ranking (<5 minutes)

```
candidate_embeddings.npy + candidates.jsonl
      │
      ├──► FAISS IndexFlatL2       ← fast vector search, returns top 5000 candidates
      │
      │    JD text ──► embed ──► FAISS query
      │
      ├──► jd_scorer.py            ← explicit skill match (45% weight)
      │       • must-have / nice-to-have keywords
      │       • role disqualifier penalties
      │       • consulting-only career penalty
      │
      ├──► heuristics.py           ← behavioral multiplier (20% weight)
      │       • response rate, interview completion, recency
      │       • honeypot detection (3 checks)
      │
      ├──► ranking.py              ← combines all 3 signals → final_score
      │
      └──► reasoning.py            ← factual per-candidate reasoning (no hallucination)
      │
      ▼
team_VertexLabs.csv (top 100 candidates)
```

### Scoring Formula

```
final_score = (0.45 × jd_skill_score)
            + (0.35 × semantic_score)
            + (0.20 × behavioral_multiplier)
```

### Honeypot Detection

Three checks that flag impossible profiles:
1. **Expert + 0 months**: ≥10 skills marked `expert` with `duration_months=0`
2. **YoE impossibility**: Stated experience >15 years beyond the earliest career date
3. **Perfect assessments**: ≥5 Redrob scores all ≥98/100

---

## 🛡️ Key Design Decisions

**Why not use only FAISS semantic search?**  
Semantic similarity alone is fooled by keyword stuffers. A "Marketing Manager" who mentions AI tools in their summary scores high on cosine similarity to an AI Engineer JD. The explicit JD skill matcher (45% weight) catches this and penalises wrong-domain titles.

**Why split pre-computation from ranking?**  
Generating embeddings for 100K candidates takes ~40 minutes on CPU. The spec allows pre-computation to exceed 5 minutes — only the *ranking* step must finish within the budget. With pre-computed embeddings, the online step (FAISS + scoring + reasoning) runs in ~30-60 seconds.

**Why template-based reasoning instead of an LLM?**  
The spec disallows external API calls during ranking. Our reasoning engine injects only verified facts from each candidate's JSON profile into structured templates — zero hallucination risk.

---

## 📊 Notebooks

The `notebooks/` folder documents the development process:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Data loading, schema inspection |
| `02_feature_engineering.ipynb` | Candidate stringifier design |
| `03_baseline_faiss.ipynb` | Baseline semantic search, spotting the keyword trap |
| `04_advanced_ranking.ipynb` | Full 3-component pipeline with behavioral re-ranking |

---

## 🔧 Compute Environment

- **Platform**: Windows 11, Intel CPU
- **Python**: 3.11
- **RAM**: 16 GB
- **GPU**: Not used (CPU-only)
- **Network during ranking**: None (all models cached locally)

---

## 📄 License

This submission is original work created for the Redrob India Runs Hackathon 2026.
