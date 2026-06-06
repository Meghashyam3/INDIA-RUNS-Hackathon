import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import re
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from data_ingestion import load_candidates
from features import stringify_candidate
from embeddings import EmbeddingModel, CandidateFAISS
from ranking import calculate_final_scores
from reasoning import generate_reasoning

# =========================================
# 1. DESIGN SYSTEM & VISUAL CONSTANTS
# =========================================
st.set_page_config(layout="wide", page_title="Redrob Ranker Sandbox")

st.markdown("""
<style>
    /* Force primary brand colors and dark slate body */
    .stApp {
        background-color: #FAFAFA;
        color: #111827;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Primary Button Styling */
    div.stButton {
        width: 100%;
    }
    div.stButton > button:first-child {
        background-color: #605DEC !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #4f4bdb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(96, 93, 236, 0.4) !important;
    }
    
    /* Download Button Override */
    div.stDownloadButton {
        width: 100%;
    }
    div.stDownloadButton > button:first-child {
        background-color: #605DEC !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    div.stDownloadButton > button:first-child:hover {
        background-color: #4f4bdb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(96, 93, 236, 0.4) !important;
    }

    /* File uploader — right-align the upload button within each column */
    div[data-testid="stFileUploader"] {
        width: 100% !important;
    }
    div[data-testid="stFileUploader"] section,
    section[data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
        gap: 0.75rem !important;
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Clean layout adjustments */
    .block-container {
        max-width: 1100px !important;
        padding-top: 3rem !important;
    }
    
    /* Native Details Expander Styling for professional look */
    details.candidate-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        background: #FFFFFF;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease-in-out;
    }
    details.candidate-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    details.candidate-card > summary {
        padding: 1.25rem 1.5rem;
        cursor: pointer;
        list-style: none; /* Hide default arrow */
    }
    details.candidate-card > summary::-webkit-details-marker {
        display: none;
    }
    details.candidate-card[open] > summary {
        border-bottom: 1px solid #F3F4F6;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# CACHED BACKEND PIPELINE
# =========================================
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    return EmbeddingModel('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=True)
def run_ranking_pipeline(candidates, jd_text, _embeddings_array):
    embedder = load_embedding_model()
    jd_embedding = embedder.generate_embeddings([jd_text])[0]
    
    faiss_index = CandidateFAISS(embedding_dim=384)
    faiss_index.add_embeddings(_embeddings_array)
    
    # Retrieve top-K semantically closest — use all candidates in demo mode
    # (rank.py uses 5000 for the full 100K+ pool; here dataset is small)
    k_to_retrieve = len(candidates)  # retrieve everything in sandbox (it's ≤500)
    distances, indices = faiss_index.search(jd_embedding, k=k_to_retrieve)
    
    final_rankings = calculate_final_scores(candidates, distances, indices)
    
    # Record how many passed semantic stage before we truncate
    semantic_passed_count = len(final_rankings)
    
    top_n = min(100, len(final_rankings))
    final_rankings = final_rankings[:top_n]
    
    final_rankings.sort(key=lambda r: (-r['final_score'], r['candidate'].get('candidate_id', '')))
    
    for i, res in enumerate(final_rankings):
        res['rank'] = i + 1
        res['reasoning'] = generate_reasoning(
            res['candidate'], res['rank'], res['behavioral_result'], res['jd_result']
        )
    
    # Attach semantic_passed_count so the caller can display it
    final_rankings[0]['_semantic_passed'] = semantic_passed_count if final_rankings else 0
        
    return final_rankings

# =========================================
# 2. THE INGESTION ZONE (TOP CENTER)
# =========================================
st.markdown("<h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem;'>HR Manager Talent Overview</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.1rem; margin-bottom: 3rem;'>AI-Driven Screening with Integrity Check & Candidate Comparison</p>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("<h2 style='text-align: center; color: #111827; margin-top: 0; border-bottom: 1px dashed #E5E7EB; padding-bottom: 1rem; margin-bottom: 1.5rem;'>Upload Center</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 10px;'>Upload Candidate Database</div>", unsafe_allow_html=True)
        uploaded_candidates = st.file_uploader("Upload Candidate Database (.json, .jsonl)", type=["json", "jsonl"], label_visibility="collapsed")
    with col2:
        st.markdown("<div style='text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 10px;'>Upload Job Description</div>", unsafe_allow_html=True)
        uploaded_jd = st.file_uploader("Upload Job Description (.txt, .docx)", type=["txt", "docx"], label_visibility="collapsed")
            
st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)

# ---- Analyze button: columns [3,2,3] forces a compact centered button ----
_, btn_col, _ = st.columns([3, 2, 3])
with btn_col:
    submit_clicked = st.button("🚀 Analyze", use_container_width=True)

# =========================================
# EXECUTION & RESULTS
# =========================================
if submit_clicked:
    with st.spinner("Executing semantic search and checking behavioral multipliers..."):
        time.sleep(0.5)
        
        candidates = []
        embeddings_array = None
        jd_text = ""
        
        try:
            if uploaded_candidates is None or uploaded_jd is None:
                st.error("⚠️ Missing Files: Please upload both Candidate and JD files.")
                st.stop()
            
            jd_text = uploaded_jd.read().decode('utf-8')
            cand_content = uploaded_candidates.read().decode('utf-8')
                
            if uploaded_candidates.name.endswith('.jsonl'):
                candidates = [json.loads(line) for line in cand_content.splitlines() if line.strip()]
            else:
                data = json.loads(cand_content)
                candidates = data if isinstance(data, list) else [data]
            
            if len(candidates) > 500:
                st.info(
                    f"**✓ Hackathon Spec 10.5 Compliant**: Large dataset detected ({len(candidates):,} candidates). "
                    "Per Rule 10.5, the sandbox safely evaluates a sample subset (500) to ensure lightning-fast UI performance "
                    "within the compute budget. To evaluate the full 100K+ pool, run the offline pipeline (`src/rank.py`)."
                )
                candidates = candidates[:500]
            
            embedder = load_embedding_model()
            candidate_strings = [stringify_candidate(c) for c in candidates]
            embeddings_array = embedder.generate_embeddings(candidate_strings)
                    
        except Exception as e:
            st.error(f"Ingestion Error: {e}")
            st.stop()
            
        results = run_ranking_pipeline(candidates, jd_text, embeddings_array)
        
    # =========================================
    # 3. PERFORMANCE METRICS BANNER
    # =========================================
    total_candidates = len(candidates)
    # Use the accurate FAISS-passed count stored by the pipeline
    semantic_passed = results[0].get('_semantic_passed', total_candidates) if results else total_candidates
    traps_blocked = sum(
        1 for r in results
        if r['behavioral_result'].get('is_honeypot')
        or any('current_role_is' in d or 'entire_career' in d
               for d in r['jd_result'].get('disqualifiers', []))
    )
    
    st.markdown("<h2 style='text-align: center; margin-top: 2rem; margin-bottom: 2rem;'>Redrob Talent Radar - Candidate Comparison & Integrity Feed</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 2rem; width: 100%; margin: 0 auto 2.5rem auto; max-width: 900px;">
        <div style="text-align: center; border: 1px solid #E5E7EB; border-radius: 12px; background: #FFFFFF; padding: 1.5rem 1rem; flex: 1; max-width: 260px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);">
            <div style="font-size: 14px; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Total Candidates Scored</div>
            <div style="font-size: 42px; color: #111827; font-weight: 700; line-height: 1;">{total_candidates:,}</div>
        </div>
        <div style="text-align: center; border: 1px solid #E5E7EB; border-radius: 12px; background: #FFFFFF; padding: 1.5rem 1rem; flex: 1; max-width: 260px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);">
            <div style="font-size: 14px; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Semantic Matches Passed</div>
            <div style="font-size: 42px; color: #111827; font-weight: 700; line-height: 1;">{semantic_passed:,}</div>
        </div>
        <div style="text-align: center; border: 1px solid #E5E7EB; border-radius: 12px; background: #FFFFFF; padding: 1.5rem 1rem; flex: 1; max-width: 260px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);">
            <div style="font-size: 14px; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Active Traps Blocked</div>
            <div style="font-size: 42px; color: #605DEC; font-weight: 700; line-height: 1;">{traps_blocked:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><hr style='border: none; border-top: 1px solid #E5E7EB; margin-bottom: 2rem;'><br>", unsafe_allow_html=True)

    # =========================================
    # 4. REFACTORED CENTRAL FEED & PROFILE CARDS
    # =========================================
    for r in results:
        c = r['candidate']
        prof = c.get('profile', {})
        hist = c.get('career_history', [])
        
        c_id = c.get('candidate_id', 'Unknown')
        c_rank = r['rank']
        c_score = r['final_score'] * 100  # Convert to percentage for display
        c_title = hist[0].get('title', 'Unknown Title') if hist else prof.get('headline', 'Unknown Title')
        c_yoe = prof.get('years_of_experience', 0)
        
        beh_res = r['behavioral_result']
        jd_res = r['jd_result']
        
        # Defensive Warning Flags
        has_hard_disqualifier = any(d for d in jd_res.get('disqualifiers', []) if "current_role_is" in d or "entire_career" in d)
        
        warning_html = ""
        pill_color = "#D1FAE5"  # Green
        pill_text = "#065F46"
        pill_label = "Best Fit"
        
        if beh_res.get('is_honeypot') or has_hard_disqualifier:
            warning_html = """
            <div style="margin-top: 8px; color: #B91C1C; background-color: #FEE2E2; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block;">
                ⚠️ Defensive Flag: Keyword Stuffer/Wrong Domain
            </div>
            """
            pill_color = "#FEE2E2" # Red
            pill_text = "#B91C1C"
            pill_label = "Low Match"
        
        raw_card = f"""
        <details class="candidate-card" {'open' if c_rank <= 3 else ''}>
            <summary>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 16px; color: #111827; font-weight: 500; width: 100%;">
                        <span style="color: #605DEC; font-weight: 800; margin-right: 12px; font-size: 18px;">Rank #{c_rank}</span>
                        <span style="font-weight: 600; color: #111827; font-size: 17px;">Candidate ID: {c_id}</span> | {c_title} | {c_yoe} YoE
                        <span style="background-color: {pill_color}; color: {pill_text}; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 13px; margin-left: 12px; float: right;">{pill_label}: {c_score:.1f}%</span>
                    </div>
                </div>
                {warning_html}
            </summary>
            <div style="padding: 1.5rem; background-color: #FAFAFA; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;">
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-bottom: 6px;">AI-Generated Reasoning:</div>
                <div style="font-size: 14px; color: #4B5563; margin-bottom: 20px; line-height: 1.6;">{r['reasoning']}</div>
                
                <div style="font-size: 14px; font-weight: 700; color: #111827; margin-bottom: 10px;">Sub-Scores:</div>
                <div style="display: flex; gap: 48px; font-size: 14px; color: #4B5563;">
                    <div><span style="font-weight: 600; color:#374151;">Semantic Match:</span> {r['semantic_score']*100:.1f}%</div>
                    <div><span style="font-weight: 600; color:#374151;">Keyword Density:</span> {jd_res['score']*100:.1f}%</div>
                    <div><span style="font-weight: 600; color:#374151;">Behavioral Indicators:</span> {beh_res['multiplier']*100:.1f}%</div>
                </div>
            </div>
        </details>
        """
        
        clean_card = re.sub(r"^[ \t]+", "", raw_card, flags=re.MULTILINE)
        st.markdown(clean_card, unsafe_allow_html=True)

    # =========================================
    # 5. EXPORT ANCHOR (BOTTOM CENTER)
    # =========================================
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    csv_data = []
    for r in results:
        csv_data.append({
            'candidate_id': r['candidate'].get('candidate_id'),
            'rank': r['rank'],
            'score': round(r['final_score'], 4),
            'reasoning': r['reasoning']
        })
    
    # Strictly exact 100 rows, strictly these columns
    csv_df = pd.DataFrame(csv_data, columns=['candidate_id', 'rank', 'score', 'reasoning']).head(100)
    csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
    
    # ---- Export button: same column trick for guaranteed centering ----
    _, exp_col, _ = st.columns([3, 2, 3])
    with exp_col:
        st.download_button(
            label="📥 Export Report",
            data=csv_bytes,
            file_name="final_evaluations.csv",
            mime="text/csv",
            use_container_width=True
        )
