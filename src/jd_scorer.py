"""
JD Skill Matcher - Explicitly scores candidates against the specific requirements
extracted from the Job Description text.

This is the most important scoring component because it captures INTENT, not just 
semantic similarity. A "Marketing Manager" can accidentally have high semantic 
similarity to an AI JD if their profile mentions tools, but will score 0 here.
"""
from typing import Dict, Any, List

# ============================================================
# HARD-CODED JD SIGNAL SETS (parsed from job_description.txt)
# ============================================================

# Tier 1: Must-have skills mentioned explicitly in the JD
MUST_HAVE_SKILLS = {
    'embeddings', 'sentence-transformers', 'sentence transformers', 'faiss',
    'vector database', 'vector search', 'pinecone', 'weaviate', 'qdrant',
    'milvus', 'opensearch', 'elasticsearch', 'retrieval', 'ranking', 'llm',
    'large language model', 'python', 'ndcg', 'mrr', 'map', 'evaluation',
    'bge', 'e5', 'hybrid search', 'dense retrieval', 'semantic search',
    'nlp', 'natural language processing', 'information retrieval',
    'recommendation', 'reranking', 're-ranking'
}

# Tier 2: Good-to-have skills mentioned in the JD
NICE_TO_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'finetuning',
    'xgboost', 'learning to rank', 'learning-to-rank', 'rag',
    'retrieval augmented', 'langchain', 'distributed systems',
    'open source', 'a/b testing', 'mlops', 'pytorch', 'transformers',
    'huggingface', 'hugging face', 'openai', 'bert', 'inference optimization',
    'data engineering', 'spark', 'kafka', 'airflow', 'mlflow', 'kubeflow'
}

# Red-flag signals: These are EXPLICIT disqualifiers mentioned in the JD
DISQUALIFIER_COMPANIES = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 
    'cognizant', 'capgemini', 'hcl technologies', 'tech mahindra'
}

# Hard role disqualifiers - clearly wrong domain, large penalty
ROLE_DISQUALIFIERS = {
    'marketing manager', 'hr manager', 'accountant', 'civil engineer',
    'mechanical engineer', 'graphic designer', 'customer support',
    'sales manager', 'finance manager', 'administrative'
}

# Soft role disqualifiers - adjacent but clearly not the target profile, smaller penalty
ROLE_SOFT_DISQUALIFIERS = {
    'project manager', 'operations manager', 'business analyst',
    'cloud engineer', 'full stack', 'frontend engineer',
    'java developer', '.net developer', 'devops engineer',
    'mobile developer', 'qa engineer'
}

# Ideal role titles — roles that are genuinely right for a Senior AI Engineer position
IDEAL_TITLES = {
    'ai engineer', 'ml engineer', 'machine learning engineer',
    'data scientist', 'nlp engineer', 'search engineer', 'ranking engineer',
    'recommendation engineer', 'recommendation systems engineer',
    'applied scientist', 'research engineer', 'applied ml engineer',
    'software engineer', 'backend engineer', 'data engineer',
    'platform engineer', 'senior engineer', 'staff engineer',
    'retrieval engineer', 'applied ai', 'senior software engineer',
}


def compute_jd_skill_score(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with:
      - 'score': float 0.0 to 1.0
      - 'matched_must_have': list of matched must-have skills
      - 'matched_nice_to_have': list of matched nice-to-have skills
      - 'disqualifiers': list of triggered disqualifier reasons
      - 'title_match': bool
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    certifications = candidate.get('certifications', [])
    signals = candidate.get('redrob_signals', {})

    # Build a flat text of all candidate skill names (lowercase)
    candidate_skill_names = {s.get('name', '').lower() for s in skills}
    
    # Build full text corpus for keyword matching
    full_text_parts = []
    full_text_parts.append(profile.get('headline', '').lower())
    full_text_parts.append(profile.get('summary', '').lower())
    full_text_parts.append(profile.get('current_title', '').lower())
    for role in career:
        full_text_parts.append(role.get('title', '').lower())
        full_text_parts.append(role.get('description', '').lower())
        full_text_parts.append(role.get('company', '').lower())
    for cert in certifications:
        full_text_parts.append(cert.get('name', '').lower())
    for skill_name in candidate_skill_names:
        full_text_parts.append(skill_name)
    
    full_text = ' '.join(full_text_parts)
    
    # --- Score Must-Have Skills ---
    matched_must_have = []
    for skill in MUST_HAVE_SKILLS:
        if skill in full_text:
            matched_must_have.append(skill)
    
    # --- Score Nice-to-Have Skills ---
    matched_nice_to_have = []
    for skill in NICE_TO_HAVE_SKILLS:
        if skill in full_text:
            matched_nice_to_have.append(skill)
    
    # --- Check Disqualifiers ---
    disqualifiers = []
    
    # 1. Company disqualifiers (only if their ENTIRE career is at consulting firms)
    all_companies = [role.get('company', '').lower() for role in career]
    consulting_count = sum(
        1 for c in all_companies 
        if any(dc in c for dc in DISQUALIFIER_COMPANIES)
    )
    if len(all_companies) > 0 and consulting_count == len(all_companies):
        disqualifiers.append("entire_career_at_consulting_firms")
    
    # 2. Role disqualifiers (current title)
    current_title = profile.get('current_title', '').lower()
    for bad_title in ROLE_DISQUALIFIERS:
        if bad_title in current_title:
            disqualifiers.append(f"current_role_is_{bad_title.replace(' ', '_')}")
            break
    
    # 2b. Soft role disqualifiers (partial penalty)
    for soft_title in ROLE_SOFT_DISQUALIFIERS:
        if soft_title in current_title:
            disqualifiers.append(f"soft_role_mismatch_{soft_title.replace(' ', '_')}")
            break
    
    # 3. YoE range (JD says 5-9, but not a hard cutoff)
    yoe = profile.get('years_of_experience', 0)
    if yoe < 2:
        disqualifiers.append("too_junior_under_2_years")
    
    # --- Title Match ---
    title_match = any(t in current_title for t in IDEAL_TITLES)

    # --- Compute Final Score ---
    # Must-have: each match worth 10 points (max 100), then normalize
    must_have_score = min(1.0, len(matched_must_have) / 6.0)  # 6 matches = perfect
    
    # Nice-to-have: each match worth bonus, normalized
    nice_to_have_score = min(1.0, len(matched_nice_to_have) / 4.0)  # 4 matches = max bonus
    
    # Title bonus
    title_bonus = 0.15 if title_match else 0.0
    
    # Combine
    raw_score = (0.70 * must_have_score) + (0.15 * nice_to_have_score) + title_bonus
    
    # Apply disqualifier penalties
    disqualifier_penalty = 0.0
    if "entire_career_at_consulting_firms" in disqualifiers:
        disqualifier_penalty += 0.4
    if any("current_role_is" in d for d in disqualifiers):
        disqualifier_penalty += 0.5   # hard penalty for clearly wrong domain
    if any("soft_role_mismatch" in d for d in disqualifiers):
        # Tier-1 soft: adjacent-to-engineering (project manager, ops manager) — small nudge
        tier1_soft = {'project_manager', 'operations_manager'}
        if any(any(t in d for t in tier1_soft) for d in disqualifiers):
            disqualifier_penalty += 0.15
        else:
            # Tier-2 soft: clearly wrong engineering domain (frontend, java, .net, devops, qa, mobile)
            # These roles are software engineers but NOT AI/ML engineers — larger penalty
            disqualifier_penalty += 0.28
    if "too_junior_under_2_years" in disqualifiers:
        disqualifier_penalty += 0.2

    final_score = max(0.0, raw_score - disqualifier_penalty)
    
    return {
        'score': round(final_score, 4),
        'matched_must_have': matched_must_have,
        'matched_nice_to_have': matched_nice_to_have,
        'disqualifiers': disqualifiers,
        'title_match': title_match
    }
