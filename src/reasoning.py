"""
Reasoning Generator

Generates a factual 1-2 sentence justification for each candidate's ranking.
Every claim is pulled directly from the candidate's JSON — no hallucination.
"""


def generate_reasoning(candidate, rank, behavioral_result, jd_result):
    """
    Returns a unique, rank-appropriate reasoning string for the candidate.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    education = candidate.get('education', [])

    title = profile.get('current_title', 'Professional')
    yoe = profile.get('years_of_experience', 0)
    company = profile.get('current_company', 'current employer')
    location = profile.get('location', '')

    # Top skills by endorsement count
    top_skills = [s.get('name') for s in sorted(skills, key=lambda s: s.get('endorsements', 0), reverse=True)[:3] if s.get('name')]

    # Behavioral signals
    notice_days = signals.get('notice_period_days', 60)
    github = signals.get('github_activity_score', -1)
    interview_rate = signals.get('interview_completion_rate', 1.0)

    # Education
    edu_str = ""
    if education:
        edu = education[0]
        degree = edu.get('degree', '')
        field = edu.get('field_of_study', '')
        tier = edu.get('tier', '')
        if degree and field:
            edu_str = f"{degree} in {field}"
            if tier in ('tier_1', 'tier_2'):
                edu_str += f" ({tier.replace('_', ' ')} institution)"

    matched_must = jd_result.get('matched_must_have', [])
    matched_nice = jd_result.get('matched_nice_to_have', [])
    disqualifiers = jd_result.get('disqualifiers', [])
    title_match = jd_result.get('title_match', False)
    is_honeypot = behavioral_result.get('is_honeypot', False)
    behavioral_notes = behavioral_result.get('behavioral_notes', [])

    # Honeypots get a short, honest note
    if is_honeypot:
        return (
            f"{title} with {yoe:.0f} years of experience; "
            f"profile flagged with statistically inconsistent signals and ranked out of consideration."
        )

    # --- Sentence 1: Lead with the most relevant fact ---
    if title_match and matched_must:
        s1 = f"{title} at {company} ({yoe:.0f} yrs) with verified signals in {', '.join(matched_must[:3])}"
    elif matched_must:
        s1 = f"{title} ({yoe:.0f} yrs) with relevant background in {', '.join(matched_must[:3])}"
    elif top_skills:
        s1 = f"{title} at {company} ({yoe:.0f} yrs); top skills include {', '.join(top_skills[:2])}"
    else:
        s1 = f"{title} with {yoe:.0f} years of total experience"

    if edu_str and rank <= 30:
        s1 += f"; holds {edu_str}"

    preferred_cities = ['pune', 'noida', 'delhi', 'mumbai', 'hyderabad', 'bangalore', 'bengaluru']
    if location and any(city in location.lower() for city in preferred_cities):
        s1 += f"; located in {location}"

    sentence1 = s1 + "."

    # --- Sentence 2: Rank-consistent commentary ---
    parts = []

    if rank <= 10:
        if matched_must:
            parts.append(f"Strong JD alignment: {len(matched_must)} of the must-have signals matched")
        if github > 60:
            parts.append(f", with active open-source presence (GitHub: {github:.0f}/100)")
        if notice_days <= 30:
            parts.append(f"; immediately available ({notice_days}-day notice)")
        elif notice_days <= 60:
            parts.append(f"; manageable notice period ({notice_days} days)")
        critical = next((n for n in behavioral_notes if any(w in n for w in ['very low', 'critical', 'inactive for'])), None)
        if critical:
            parts.append(f"; note: {critical}")

    elif rank <= 30:
        if matched_must:
            parts.append(f"Matches {len(matched_must)} of the JD's core technical signals")
        concern = next((n for n in behavioral_notes if any(w in n for w in ['very low', 'below-average', 'low recent', 'inactive', 'critical', 'long notice'])), None)
        if concern:
            parts.append(f"; concern: {concern}")
        elif notice_days > 90:
            parts.append(f"; note: long notice period ({notice_days} days) may delay start")
        elif not matched_must:
            parts.append("; limited direct skill overlap with JD but strong behavioral profile")

    elif rank <= 70:
        if disqualifiers:
            parts.append(f"Ranked mid-tier due to profile gaps; {disqualifiers[0].replace('_', ' ')}")
        elif not matched_must:
            parts.append("Limited overlap with core JD technical requirements")
        else:
            parts.append(f"Partial technical alignment ({len(matched_must)} must-have signals) but edged out by stronger profiles")
        concern = next((n for n in behavioral_notes if any(w in n for w in ['low', 'inactive', 'critical'])), None)
        if concern:
            parts.append(f"; {concern}")

    else:
        if disqualifiers:
            parts.append(f"Ranked lower due to {disqualifiers[0].replace('_', ' ')}")
        elif not matched_must:
            parts.append("Minimal overlap with core JD requirements; included as boundary filler")
        else:
            parts.append("Below threshold on combined technical and behavioral scoring")
        if interview_rate < 0.5:
            parts.append(f"; critical flag: {interview_rate:.0%} interview completion rate")

    if parts:
        return f"{sentence1} {''.join(parts)}."
    return sentence1
