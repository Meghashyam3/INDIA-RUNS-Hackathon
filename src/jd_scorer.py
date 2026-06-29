"""
JD Skill Scorer

Scores a candidate against the specific requirements from the job description.
Uses must-have skills, nice-to-have skills, role titles, and disqualifiers.
"""

# Must-have skills from the JD
MUST_HAVE_SKILLS = {
    'embeddings', 'sentence-transformers', 'sentence transformers', 'faiss',
    'vector database', 'vector search', 'pinecone', 'weaviate', 'qdrant',
    'milvus', 'opensearch', 'elasticsearch', 'retrieval', 'ranking', 'llm',
    'large language model', 'python', 'ndcg', 'mrr', 'map', 'evaluation',
    'bge', 'e5', 'hybrid search', 'dense retrieval', 'semantic search',
    'nlp', 'natural language processing', 'information retrieval',
    'recommendation', 'reranking', 're-ranking'
}

# Nice-to-have skills from the JD
NICE_TO_HAVE_SKILLS = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'finetuning',
    'xgboost', 'learning to rank', 'learning-to-rank', 'rag',
    'retrieval augmented', 'langchain', 'distributed systems',
    'open source', 'a/b testing', 'mlops', 'pytorch', 'transformers',
    'huggingface', 'hugging face', 'openai', 'bert', 'inference optimization',
    'data engineering', 'spark', 'kafka', 'airflow', 'mlflow', 'kubeflow'
}

# Companies that disqualify if a candidate's entire career is there
DISQUALIFIER_COMPANIES = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl technologies', 'tech mahindra'
}

# Clearly wrong domain roles — heavy penalty
ROLE_DISQUALIFIERS = {
    'marketing manager', 'hr manager', 'accountant', 'civil engineer',
    'mechanical engineer', 'graphic designer', 'customer support',
    'sales manager', 'finance manager', 'administrative'
}

# Adjacent roles — light penalty
ROLE_SOFT_DISQUALIFIERS = {
    'project manager', 'operations manager', 'business analyst',
    'cloud engineer', 'full stack', 'frontend engineer',
    'java developer', '.net developer', 'devops engineer',
    'mobile developer', 'qa engineer'
}

# Roles that are a genuine match for this JD
IDEAL_TITLES = {
    'ai engineer', 'ml engineer', 'machine learning engineer',
    'data scientist', 'nlp engineer', 'search engineer', 'ranking engineer',
    'recommendation engineer', 'recommendation systems engineer',
    'applied scientist', 'research engineer', 'applied ml engineer',
    'software engineer', 'backend engineer', 'data engineer',
    'platform engineer', 'senior engineer', 'staff engineer',
    'retrieval engineer', 'applied ai', 'senior software engineer',
}


def compute_jd_skill_score(candidate):
    """
    Returns a dict with score (0.0 to 1.0), matched skills, and disqualifiers.
    """
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    certifications = candidate.get('certifications', [])

    candidate_skill_names = {s.get('name', '').lower() for s in skills}

    # Build one big text blob from all candidate fields for keyword matching
    text_pieces = (
        [profile.get('headline', ''), profile.get('summary', ''), profile.get('current_title', '')]
        + [role.get('title', '') + ' ' + role.get('description', '') + ' ' + role.get('company', '') for role in career]
        + [cert.get('name', '') for cert in certifications]
        + list(candidate_skill_names)
    )
    full_text = ' '.join(text_pieces).lower()

    # Match must-have and nice-to-have skills
    matched_must = [s for s in MUST_HAVE_SKILLS if s in full_text]
    matched_nice = [s for s in NICE_TO_HAVE_SKILLS if s in full_text]

    # Check disqualifiers
    disqualifiers = []

    all_companies = [role.get('company', '').lower() for role in career]
    if all_companies and all(any(dc in c for dc in DISQUALIFIER_COMPANIES) for c in all_companies):
        disqualifiers.append("entire_career_at_consulting_firms")

    current_title = profile.get('current_title', '').lower()
    for bad_title in ROLE_DISQUALIFIERS:
        if bad_title in current_title:
            disqualifiers.append(f"current_role_is_{bad_title.replace(' ', '_')}")
            break

    for soft_title in ROLE_SOFT_DISQUALIFIERS:
        if soft_title in current_title:
            disqualifiers.append(f"soft_role_mismatch_{soft_title.replace(' ', '_')}")
            break

    yoe = profile.get('years_of_experience', 0)
    if yoe < 2:
        disqualifiers.append("too_junior_under_2_years")

    title_match = any(t in current_title for t in IDEAL_TITLES)

    # Compute score
    must_score = min(1.0, len(matched_must) / 6.0)
    nice_score = min(1.0, len(matched_nice) / 4.0)
    title_bonus = 0.15 if title_match else 0.0
    raw_score = (0.70 * must_score) + (0.15 * nice_score) + title_bonus

    # Apply penalties
    penalty = 0.0
    if "entire_career_at_consulting_firms" in disqualifiers:
        penalty += 0.4
    if any("current_role_is" in d for d in disqualifiers):
        penalty += 0.5
    if any("soft_role_mismatch" in d for d in disqualifiers):
        tier1 = {'project_manager', 'operations_manager'}
        penalty += 0.15 if any(any(t in d for t in tier1) for d in disqualifiers) else 0.28
    if "too_junior_under_2_years" in disqualifiers:
        penalty += 0.2

    final_score = max(0.0, raw_score - penalty)

    return {
        'score': round(final_score, 4),
        'matched_must_have': matched_must,
        'matched_nice_to_have': matched_nice,
        'disqualifiers': disqualifiers,
        'title_match': title_match
    }
