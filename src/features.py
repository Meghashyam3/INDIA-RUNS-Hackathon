def stringify_candidate(candidate):
    """
    Converts a candidate dict into a single text string for embedding.
    Pulls title, summary, skills, career history, and education.
    """
    parts = []

    profile = candidate.get('profile', {})
    if profile.get('current_title'):
        parts.append(f"Title: {profile['current_title']}")
    if profile.get('summary'):
        parts.append(f"Summary: {profile['summary']}")

    skills = candidate.get('skills', [])
    if skills:
        skill_list = []
        for s in skills:
            name = s.get('name', '')
            prof = s.get('proficiency', '')
            if name:
                skill_list.append(f"{name} ({prof})" if prof else name)
        parts.append("Skills: " + ", ".join(skill_list))

    career = candidate.get('career_history', [])
    if career:
        career_list = []
        for role in career:
            role_title = role.get('title', '')
            desc = role.get('description', '').replace('\n', ' ')
            if role_title:
                career_list.append(f"Role: {role_title} - {desc}" if desc else f"Role: {role_title}")
        parts.append("Experience: " + " | ".join(career_list))

    education = candidate.get('education', [])
    if education:
        edu_list = []
        for edu in education:
            degree = edu.get('degree', '')
            field = edu.get('field_of_study', '')
            if degree or field:
                edu_list.append(f"{degree} in {field}".strip())
        parts.append("Education: " + ", ".join(edu_list))

    return ". ".join(parts)
