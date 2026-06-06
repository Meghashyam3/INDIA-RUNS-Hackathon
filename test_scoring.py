import sys
import json
sys.path.append('src')
from jd_scorer import compute_jd_skill_score

# Test the candidates that were wrongly ranked high
test_cases = [
    {'candidate_id': 'CAND_0000021', 'label': 'Project Manager (rank 3 old)', 'profile': {'current_title': 'Project Manager', 'years_of_experience': 14}, 'skills': [{'name': 'faiss'}, {'name': 'vector database'}, {'name': 'recommendation'}], 'career_history': [{'title': 'Project Manager', 'company': 'TechCo', 'description': 'managed ai projects'}], 'certifications': [], 'redrob_signals': {}},
    {'candidate_id': 'CAND_0000032', 'label': '.NET Developer at Cognizant (rank 5 old)', 'profile': {'current_title': '.NET Developer', 'years_of_experience': 8}, 'skills': [{'name': 'python'}, {'name': 'embeddings'}], 'career_history': [{'title': '.NET Developer', 'company': 'Cognizant', 'description': 'backend work'}], 'certifications': [], 'redrob_signals': {}},
    {'candidate_id': 'CAND_0000014', 'label': 'Frontend Engineer at Zomato (rank 6 old)', 'profile': {'current_title': 'Frontend Engineer', 'years_of_experience': 8}, 'skills': [{'name': 'faiss'}, {'name': 'opensearch'}], 'career_history': [{'title': 'Frontend Engineer', 'company': 'Zomato', 'description': 'ui work'}], 'certifications': [], 'redrob_signals': {}},
    {'candidate_id': 'CAND_0000001', 'label': 'AI Engineer (good)', 'profile': {'current_title': 'AI Engineer', 'years_of_experience': 6}, 'skills': [{'name': 'faiss'}, {'name': 'embeddings'}, {'name': 'python'}, {'name': 'retrieval'}], 'career_history': [{'title': 'AI Engineer', 'company': 'Startupco', 'description': 'built ranking systems'}], 'certifications': [], 'redrob_signals': {}},
]
print('SCORING AFTER FIX:')
print('-'*70)
for c in test_cases:
    result = compute_jd_skill_score(c)
    print(f"{c['label']}:\n  score={result['score']}\n  disqualifiers={result['disqualifiers']}\n  title_match={result['title_match']}\n")
