from typing import Dict, Any

def stringify_candidate(candidate: Dict[str, Any]) -> str:
    """
    Converts a nested candidate dictionary into a single dense text string
    optimized for semantic embedding.
    
    Extracts:
    - Current Title & Summary
    - Skills (focusing on name and proficiency)
    - Career History (focusing on titles and descriptions)
    """
    parts = []
    
    # 1. Profile Information
    profile = candidate.get('profile', {})
    title = profile.get('current_title', '')
    summary = profile.get('summary', '')
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
        
    # 2. Skills
    skills = candidate.get('skills', [])
    if skills:
        skill_strings = []
        for s in skills:
            name = s.get('name', '')
            prof = s.get('proficiency', '')
            if name:
                skill_strings.append(f"{name} ({prof})" if prof else name)
        parts.append("Skills: " + ", ".join(skill_strings))
        
    # 3. Career History
    career = candidate.get('career_history', [])
    if career:
        career_strings = []
        for role in career:
            role_title = role.get('title', '')
            desc = role.get('description', '')
            if role_title:
                role_str = f"Role: {role_title}"
                if desc:
                    # Clean up newlines in description
                    clean_desc = desc.replace('\n', ' ')
                    role_str += f" - {clean_desc}"
                career_strings.append(role_str)
        parts.append("Experience: " + " | ".join(career_strings))
        
    # 4. Education (Optional but good for context)
    education = candidate.get('education', [])
    if education:
        edu_strings = []
        for edu in education:
            degree = edu.get('degree', '')
            field = edu.get('field_of_study', '')
            if degree or field:
                edu_strings.append(f"{degree} in {field}".strip())
        parts.append("Education: " + ", ".join(edu_strings))

    # Join everything with a clean separator
    # Using a period and space helps sentence-transformers separate concepts
    return ". ".join(parts)
