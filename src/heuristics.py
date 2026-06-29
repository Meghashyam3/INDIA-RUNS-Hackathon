"""
Behavioral scoring and honeypot detection.

Steps:
  1. Detect profiles with impossible/contradictory data (honeypots)
  2. Compute a reliability multiplier (0.0 to 1.0) from platform signals
"""
from datetime import datetime, date


def _detect_honeypot(candidate):
    """
    Flags profiles with statistically impossible signals.
    Returns (is_honeypot, reasons).
    Only flags on clearly impossible data to avoid false positives.
    """
    flags = []
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})

    # Flag 1: Expert proficiency in 10+ skills but 0 months of usage
    expert_zero = [s.get('name') for s in skills
                   if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0]
    if len(expert_zero) >= 10:
        flags.append(f"expert_in_{len(expert_zero)}_skills_with_0_months_experience")

    # Flag 2: Stated years of experience exceeds what the career dates allow
    stated_yoe = profile.get('years_of_experience', 0)
    if career:
        earliest = None
        for role in career:
            start_str = role.get('start_date')
            if start_str:
                try:
                    start = datetime.strptime(start_str, '%Y-%m-%d').date()
                    if earliest is None or start < earliest:
                        earliest = start
                except (ValueError, TypeError):
                    pass
        if earliest:
            actual_yoe = (date.today() - earliest).days / 365.25
            if stated_yoe > actual_yoe + 15:
                flags.append(f"stated_yoe_{stated_yoe:.0f}_impossible_given_career_history")

    # Flag 3: All skill assessments are suspiciously perfect (5+ scores above 98)
    scores = list(signals.get('skill_assessment_scores', {}).values())
    if len(scores) >= 5 and sum(1 for s in scores if s >= 98) >= 5:
        flags.append("suspiciously_perfect_assessment_scores")

    return len(flags) >= 1, flags


def compute_behavioral_multiplier(candidate):
    """
    Returns a dict with the behavioral multiplier and supporting info.
    Multiplier ranges from 0.0 (bad) to 1.0 (ideal).
    """
    signals = candidate.get('redrob_signals', {})
    notes = []

    is_honeypot, reasons = _detect_honeypot(candidate)
    if is_honeypot:
        return {
            'multiplier': 0.05,
            'is_honeypot': True,
            'honeypot_reasons': reasons,
            'behavioral_notes': [f'HONEYPOT: {r}' for r in reasons]
        }

    score = 1.0

    # Not open to work
    if not signals.get('open_to_work_flag', True):
        score *= 0.80
        notes.append("not currently open to work")

    # Last active date
    last_active_str = signals.get('last_active_date')
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
            days_inactive = (date.today() - last_active).days
            if days_inactive > 180:
                score *= 0.60
                notes.append(f"inactive for {days_inactive} days")
            elif days_inactive > 90:
                score *= 0.85
                notes.append(f"low recent activity ({days_inactive} days)")
        except (ValueError, TypeError):
            pass

    # Recruiter response rate
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate < 0.20:
        score *= 0.45
        notes.append(f"very low recruiter response rate ({response_rate:.0%})")
    elif response_rate < 0.50:
        score *= 0.75
        notes.append(f"below-average response rate ({response_rate:.0%})")
    else:
        notes.append(f"good recruiter response rate ({response_rate:.0%})")

    # Interview completion rate
    interview_rate = signals.get('interview_completion_rate', 1.0)
    if interview_rate < 0.40:
        score *= 0.35
        notes.append(f"critical: low interview completion rate ({interview_rate:.0%})")
    elif interview_rate < 0.70:
        score *= 0.75
        notes.append(f"moderate interview attendance ({interview_rate:.0%})")

    # Notice period
    notice_days = signals.get('notice_period_days', 60)
    if notice_days <= 30:
        notes.append(f"short notice period ({notice_days} days) — ideal")
    elif notice_days > 90:
        score *= 0.90
        notes.append(f"long notice period ({notice_days} days)")

    # Profile completeness
    completeness = signals.get('profile_completeness_score', 70) / 100.0
    if completeness < 0.50:
        score *= 0.88
        notes.append("incomplete profile")

    # GitHub activity bonus
    github = signals.get('github_activity_score', -1)
    if github > 70:
        score = min(1.0, score * 1.12)
        notes.append(f"strong GitHub activity ({github:.0f}/100)")
    elif github > 40:
        notes.append(f"moderate GitHub activity ({github:.0f}/100)")

    return {
        'multiplier': round(max(0.0, min(1.0, score)), 4),
        'is_honeypot': False,
        'honeypot_reasons': [],
        'behavioral_notes': notes
    }
