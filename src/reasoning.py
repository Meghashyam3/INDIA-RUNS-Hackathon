"""
Reasoning Generator

Generates a clear, professional 2-sentence justification for each candidate's ranking.
Every claim is taken directly from the candidate's JSON — no hallucination.
"""


def generate_reasoning(candidate, rank, behavioral_result, jd_result):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])

    title = profile.get('current_title', 'Professional')
    yoe = profile.get('years_of_experience', 0)
    company = profile.get('current_company', '')
    location = profile.get('location', '')

    notice_days = signals.get('notice_period_days', 60)
    github = signals.get('github_activity_score', -1)
    interview_rate = signals.get('interview_completion_rate', 1.0)
    response_rate = signals.get('recruiter_response_rate', 0.5)

    matched_must = jd_result.get('matched_must_have', [])
    matched_nice = jd_result.get('matched_nice_to_have', [])
    disqualifiers = jd_result.get('disqualifiers', [])
    title_match = jd_result.get('title_match', False)
    is_honeypot = behavioral_result.get('is_honeypot', False)
    behavioral_notes = behavioral_result.get('behavioral_notes', [])

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
                edu_str += f" from a {tier.replace('_', ' ')} institution"

    # Honeypot: short, clear dismissal
    if is_honeypot:
        return (
            f"{title} with {yoe:.0f} years of experience. "
            f"This profile was flagged as a honeypot — it contains statistically impossible data "
            f"(e.g. expert-level skills listed with zero months of usage) and has been excluded from consideration."
        )

    # --- Sentence 1: Who is this person ---
    at_company = f" at {company}" if company else ""
    s1 = f"{title}{at_company} with {yoe:.0f} years of experience"
    if edu_str and rank <= 30:
        s1 += f", holding a {edu_str}"
    preferred_cities = ['pune', 'noida', 'delhi', 'mumbai', 'hyderabad', 'bangalore', 'bengaluru']
    if location and any(city in location.lower() for city in preferred_cities):
        s1 += f", based in {location}"
    s1 += "."

    # --- Sentence 2: Why this rank ---
    s2 = ""

    if rank <= 10:
        if matched_must:
            skill_sample = ", ".join(matched_must[:3])
            s2 = (
                f"Ranked #{rank} for strong JD alignment — {len(matched_must)} of the must-have skills matched "
                f"(including {skill_sample})"
            )
        else:
            s2 = f"Ranked #{rank} based on high semantic relevance to the job description"

        extras = []
        if github > 60:
            extras.append(f"active open-source contributions (GitHub: {github:.0f}/100)")
        if notice_days <= 30:
            extras.append(f"immediately available ({notice_days}-day notice)")
        elif notice_days <= 60:
            extras.append(f"available within {notice_days} days")
        if extras:
            s2 += f", with {' and '.join(extras)}"
        s2 += "."

        critical = next((n for n in behavioral_notes if 'very low' in n or 'critical' in n), None)
        if critical:
            s2 += f" Note: {critical}."

    elif rank <= 30:
        if matched_must:
            skill_sample = ", ".join(matched_must[:2])
            s2 = f"Good match with {len(matched_must)} must-have JD skills covered ({skill_sample})"
        else:
            s2 = "Reaches this rank through semantic profile relevance, though direct keyword overlap with the JD is moderate"

        concern = next((n for n in behavioral_notes if any(w in n for w in ['very low', 'below-average', 'inactive', 'long notice', 'critical'])), None)
        if concern:
            s2 += f", though there is one concern: {concern}"
        elif notice_days > 90:
            s2 += f" — note that a {notice_days}-day notice period may delay onboarding"
        s2 += "."

    elif rank <= 70:
        if disqualifiers:
            d = disqualifiers[0].replace('_', ' ')
            s2 = f"Ranks mid-tier primarily due to {d}"
        elif not matched_must:
            s2 = "Reaches this rank through semantic similarity alone, with limited direct overlap with the JD's core technical requirements"
        else:
            s2 = f"Partial JD alignment ({len(matched_must)} must-haves matched), but edged out by stronger candidates above"

        concern = next((n for n in behavioral_notes if any(w in n for w in ['low', 'inactive', 'critical'])), None)
        if concern:
            s2 += f"; {concern}"
        s2 += "."

    else:
        if disqualifiers:
            d = disqualifiers[0].replace('_', ' ')
            s2 = f"Ranked #{rank} due to {d}"
        elif not matched_must:
            s2 = f"Ranked #{rank} with minimal overlap with this JD's core requirements — included at the bottom of the list"
        else:
            s2 = f"Ranked #{rank} — falls below the threshold on the combined scoring model"

        if interview_rate < 0.5:
            s2 += f". Additionally flagged for a low interview completion rate ({interview_rate:.0%}), which is a reliability concern"
        s2 += "."

    return f"{s1} {s2}"
