"""
Behavioral Heuristics & Honeypot Detector

Two jobs:
1. Compute a behavioral multiplier (0.0 - 1.0) from redrob_signals
2. Detect honeypot profiles with impossible/contradictory data and return a 
   near-zero multiplier to naturally push them out of the top 100.
"""
from typing import Dict, Any
from datetime import datetime, date


def _detect_honeypot(candidate: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Detects profiles with statistically impossible or suspicious data.
    The hackathon spec says ~80 honeypots exist with impossible profiles.
    Returns (is_honeypot: bool, reasons: list[str])

    NOTE: Be conservative. A false positive (flagging a real candidate)
    hurts NDCG@10 more than a false negative (missing a real honeypot).
    Only flag profiles with CLEARLY impossible data.
    """
    flags = []
    profile = candidate.get('profile', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})

    # --- Honeypot Check 1: Expert skill with 0 months experience ---
    # Spec explicitly mentions: "expert proficiency in 10 skills with 0 years used"
    # Threshold set HIGH (>=5) to avoid false positives.
    expert_zero_duration = [
        s.get('name') for s in skills
        if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0
    ]
    if len(expert_zero_duration) >= 10:
        flags.append(f"expert_proficiency_in_{len(expert_zero_duration)}_skills_with_0_months")

    # --- Honeypot Check 2: Stated YoE grossly exceeds career history ---
    # Spec explicitly mentions: "8 years of experience at a company founded 3 years ago"
    # Use a generous 10-year buffer to avoid false positives.
    stated_yoe = profile.get('years_of_experience', 0)
    if career:
        earliest_start = None
        for role in career:
            start_str = role.get('start_date')
            if start_str:
                try:
                    start_dt = datetime.strptime(start_str, '%Y-%m-%d').date()
                    if earliest_start is None or start_dt < earliest_start:
                        earliest_start = start_dt
                except (ValueError, TypeError):
                    pass
        if earliest_start:
            calculated_yoe = (date.today() - earliest_start).days / 365.25
            if stated_yoe > calculated_yoe + 15:
                flags.append(f"stated_yoe_{stated_yoe:.0f}_impossible_vs_career_dates_{calculated_yoe:.1f}y")

    # --- Honeypot Check 3: All skill assessments are suspiciously perfect ---
    assessment_scores = list(signals.get('skill_assessment_scores', {}).values())
    if len(assessment_scores) >= 5:
        perfect_scores = sum(1 for s in assessment_scores if s >= 98)
        if perfect_scores >= 5:
            flags.append(f"suspiciously_perfect_assessments_{perfect_scores}_scores_above_98")

    # A single flag is enough to call it a honeypot
    is_honeypot = len(flags) >= 1
    return is_honeypot, flags


def compute_behavioral_multiplier(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with:
      - 'multiplier': float 0.0 to 1.0
      - 'is_honeypot': bool
      - 'honeypot_reasons': list[str]
      - 'behavioral_notes': list[str] (for use in reasoning)
    """
    signals = candidate.get('redrob_signals', {})
    behavioral_notes = []

    # --- Step 1: Honeypot Detection ---
    is_honeypot, honeypot_reasons = _detect_honeypot(candidate)
    if is_honeypot:
        return {
            'multiplier': 0.05,  # Near-zero, not absolute zero to avoid score=0
            'is_honeypot': True,
            'honeypot_reasons': honeypot_reasons,
            'behavioral_notes': [f'HONEYPOT: {r}' for r in honeypot_reasons]
        }

    score = 1.0

    # --- Step 2: Open to Work Flag ---
    if not signals.get('open_to_work_flag', True):
        score *= 0.80
        behavioral_notes.append("not currently open to work")

    # --- Step 3: Last Active Date (recency) ---
    last_active_str = signals.get('last_active_date')
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
            days_inactive = (date.today() - last_active).days
            if days_inactive > 180:
                score *= 0.60
                behavioral_notes.append(f"inactive for {days_inactive} days")
            elif days_inactive > 90:
                score *= 0.85
                behavioral_notes.append(f"low recent activity ({days_inactive} days)")
        except (ValueError, TypeError):
            pass

    # --- Step 4: Recruiter Response Rate (anti-ghosting) ---
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate < 0.20:
        score *= 0.45
        behavioral_notes.append(f"very low recruiter response rate ({response_rate:.0%})")
    elif response_rate < 0.50:
        score *= 0.75
        behavioral_notes.append(f"below-average response rate ({response_rate:.0%})")
    else:
        behavioral_notes.append(f"good recruiter response rate ({response_rate:.0%})")

    # --- Step 5: Interview Completion Rate ---
    interview_rate = signals.get('interview_completion_rate', 1.0)
    if interview_rate < 0.40:
        score *= 0.35  # Massive penalty — ghosting interviews is a top red flag
        behavioral_notes.append(f"critical: low interview completion rate ({interview_rate:.0%})")
    elif interview_rate < 0.70:
        score *= 0.75
        behavioral_notes.append(f"moderate interview attendance ({interview_rate:.0%})")

    # --- Step 6: Notice Period (JD prefers sub-30 days) ---
    notice_days = signals.get('notice_period_days', 60)
    if notice_days <= 30:
        behavioral_notes.append(f"short notice period ({notice_days} days) — ideal")
    elif notice_days > 90:
        score *= 0.90
        behavioral_notes.append(f"long notice period ({notice_days} days)")

    # --- Step 7: Profile Completeness ---
    completeness = signals.get('profile_completeness_score', 70) / 100.0
    if completeness < 0.50:
        score *= 0.88
        behavioral_notes.append("incomplete profile")

    # --- Step 8: GitHub Activity Bonus ---
    github = signals.get('github_activity_score', -1)
    if github > 70:
        score = min(1.0, score * 1.12)
        behavioral_notes.append(f"strong GitHub activity score ({github:.0f}/100)")
    elif github > 40:
        behavioral_notes.append(f"moderate GitHub activity ({github:.0f}/100)")

    return {
        'multiplier': round(max(0.0, min(1.0, score)), 4),
        'is_honeypot': False,
        'honeypot_reasons': [],
        'behavioral_notes': behavioral_notes
    }
