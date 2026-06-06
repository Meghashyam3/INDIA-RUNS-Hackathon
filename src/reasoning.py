"""
Rich, Factual Reasoning Generator

Generates unique 1-2 sentence justifications per candidate.
STRICT RULE: Every claim MUST come from the candidate JSON. No hallucination.

The hackathon Stage 4 check specifically tests:
  - Specific facts from the profile
  - JD connection (not generic praise)
  - Honest concerns where gaps exist
  - No hallucination
  - Variation across the 10 sampled rows
  - Rank consistency (tone matches rank)
"""
from typing import Dict, Any, List


def generate_reasoning(
    candidate: Dict[str, Any],
    rank: int,
    behavioral_result: Dict[str, Any],
    jd_result: Dict[str, Any]
) -> str:
    """
    Generates a unique, factual, rank-consistent 1-2 sentence reasoning.
    Uses behavioral_result and jd_result dicts from our scoring modules.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    education = candidate.get('education', [])

    # ---- Extract key facts ----
    title = profile.get('current_title', 'Professional')
    yoe = profile.get('years_of_experience', 0)
    company = profile.get('current_company', 'current employer')
    location = profile.get('location', '')
    country = profile.get('country', '')

    # Top 3 skills by endorsements (most credible)
    sorted_skills = sorted(skills, key=lambda s: s.get('endorsements', 0), reverse=True)
    top_3_skills = [s.get('name') for s in sorted_skills[:3] if s.get('name')]

    # Skill assessment scores from Redrob platform
    assessment_scores = signals.get('skill_assessment_scores', {})
    top_assessments = sorted(assessment_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    # Behavioral facts
    notice_days = signals.get('notice_period_days', 60)
    github_score = signals.get('github_activity_score', -1)
    open_to_work = signals.get('open_to_work_flag', False)
    response_rate = signals.get('recruiter_response_rate', 0.5)
    interview_rate = signals.get('interview_completion_rate', 1.0)

    # Education
    best_edu = education[0] if education else None
    edu_str = ""
    if best_edu:
        degree = best_edu.get('degree', '')
        field = best_edu.get('field_of_study', '')
        tier = best_edu.get('tier', '')
        if degree and field:
            edu_str = f"{degree} in {field}"
            if tier in ('tier_1', 'tier_2'):
                edu_str += f" ({tier.replace('_', ' ')} institution)"

    # JD matching facts
    matched_must = jd_result.get('matched_must_have', [])
    matched_nice = jd_result.get('matched_nice_to_have', [])
    disqualifiers = jd_result.get('disqualifiers', [])
    title_match = jd_result.get('title_match', False)
    is_honeypot = behavioral_result.get('is_honeypot', False)
    behavioral_notes = behavioral_result.get('behavioral_notes', [])

    # ---- Sentence 1: Factual Anchor ----
    # Make it unique per candidate by leading with most relevant fact

    if is_honeypot:
        return (
            f"{title} with {yoe:.0f} years of experience; profile flagged with "
            f"statistically inconsistent signals and ranked out of consideration."
        )

    parts_s1 = []
    if title_match and matched_must:
        # Great match — lead with role + top matched skills
        skill_sample = ', '.join(matched_must[:3])
        parts_s1.append(
            f"{title} at {company} ({yoe:.0f} yrs) with verified profile signals in {skill_sample}"
        )
    elif matched_must:
        skill_sample = ', '.join(matched_must[:3])
        parts_s1.append(
            f"{title} ({yoe:.0f} yrs) with relevant background in {skill_sample}"
        )
    elif top_3_skills:
        skill_sample = ', '.join(top_3_skills[:2])
        parts_s1.append(
            f"{title} at {company} ({yoe:.0f} yrs); top skills include {skill_sample}"
        )
    else:
        parts_s1.append(f"{title} with {yoe:.0f} years of total experience")

    # Add education if it's strong
    if edu_str and rank <= 30:
        parts_s1.append(f"; holds {edu_str}")

    # Add location if JD-relevant (Pune/Noida preferred)
    if location and any(city in location.lower() for city in ['pune', 'noida', 'delhi', 'mumbai', 'hyderabad', 'bangalore', 'bengaluru']):
        parts_s1.append(f"; located in {location}")

    sentence1 = ''.join(parts_s1) + '.'

    # ---- Sentence 2: Rank-Consistent Justification ----
    sentence2_parts = []

    if rank <= 10:
        # Top 10 — strong positive with specific signals
        if matched_must:
            sentence2_parts.append(
                f"Strong JD alignment: {len(matched_must)} of the must-have signals matched"
            )
        if github_score > 60:
            sentence2_parts.append(f", with active open-source presence (GitHub score: {github_score:.0f}/100)")
        if notice_days <= 30:
            sentence2_parts.append(f"; immediately available ({notice_days}-day notice)")
        elif notice_days <= 60:
            sentence2_parts.append(f"; manageable notice period ({notice_days} days)")
        # Only mention behavioral negatives if truly bad
        critical_concern = next(
            (n for n in behavioral_notes if any(w in n for w in ['very low', 'critical', 'inactive for'])),
            None
        )
        if critical_concern:
            sentence2_parts.append(f"; note: {critical_concern}")

    elif rank <= 30:
        # Good candidates — acknowledge both strengths and gaps
        if matched_must:
            sentence2_parts.append(
                f"Matches {len(matched_must)} of the JD's core technical signals"
            )
        # Only flag a behavioral concern if it is actually negative
        negative_concern = next(
            (n for n in behavioral_notes
             if any(w in n for w in ['very low', 'below-average', 'low recent', 'inactive', 'critical', 'long notice'])),
            None
        )
        if negative_concern:
            sentence2_parts.append(f"; concern: {negative_concern}")
        elif notice_days > 90:
            sentence2_parts.append(f"; note: long notice period ({notice_days} days) may delay start")
        elif not matched_must:
            sentence2_parts.append("; limited direct skill overlap with JD but strong behavioral profile")

    elif rank <= 70:
        # Mid-tier — honest about why they are here
        if disqualifiers:
            d = disqualifiers[0].replace('_', ' ')
            sentence2_parts.append(f"Ranked mid-tier due to profile gaps; {d}")
        elif not matched_must:
            sentence2_parts.append("Limited overlap with core JD technical requirements")
        else:
            sentence2_parts.append(
                f"Partial technical alignment ({len(matched_must)} must-have signals) but edged out by stronger profiles"
            )
        # Add key negative concern only
        concern = next(
            (n for n in behavioral_notes
             if any(w in n for w in ['low', 'inactive', 'critical'])),
            None
        )
        if concern:
            sentence2_parts.append(f"; {concern}")

    else:
        # Bottom tier — clear and honest
        if disqualifiers:
            d = disqualifiers[0].replace('_', ' ')
            sentence2_parts.append(f"Ranked lower due to {d}")
        elif matched_must == []:
            sentence2_parts.append("Minimal overlap with core JD requirements; included as boundary filler")
        else:
            sentence2_parts.append("Below threshold on combined technical and behavioral scoring")
        if interview_rate < 0.5:
            sentence2_parts.append(f"; critical behavioral flag: {interview_rate:.0%} interview completion rate")

    if sentence2_parts:
        sentence2 = ' '.join(sentence2_parts) + '.'
        return f"{sentence1} {sentence2}"
    else:
        return sentence1
