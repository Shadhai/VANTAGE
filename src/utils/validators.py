"""Validation utilities for candidate data and submission format."""

from typing import Dict, List, Optional
from datetime import datetime


def validate_candidate_structure(candidate: Dict) -> List[str]:
    """Validate candidate has required fields. Returns list of issues."""
    issues = []
    
    required_top = ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"]
    for field in required_top:
        if field not in candidate:
            issues.append(f"Missing required field: {field}")
    
    if "profile" in candidate:
        profile = candidate["profile"]
        required_profile = [
            "anonymized_name", "headline", "summary", "location", "country",
            "years_of_experience", "current_title", "current_company"
        ]
        for field in required_profile:
            if field not in profile:
                issues.append(f"Profile missing field: {field}")
    
    if "redrob_signals" in candidate:
        signals = candidate["redrob_signals"]
        required_signals = [
            "last_active_date", "recruiter_response_rate", "open_to_work_flag",
            "interview_completion_rate", "notice_period_days"
        ]
        for field in required_signals:
            if field not in signals:
                issues.append(f"Signals missing field: {field}")
    
    return issues


def validate_submission_format(rows: List[Dict]) -> List[str]:
    """Validate submission rows match required format. Returns list of issues."""
    issues = []
    
    if len(rows) != 100:
        issues.append(f"Expected 100 rows, got {len(rows)}")
    
    ranks = set()
    ids = set()
    
    for i, row in enumerate(rows):
        row_num = i + 2  # +2 for header and 1-indexed
        
        if "candidate_id" not in row:
            issues.append(f"Row {row_num}: missing candidate_id")
        elif row["candidate_id"] in ids:
            issues.append(f"Row {row_num}: duplicate candidate_id")
        else:
            ids.add(row["candidate_id"])
        
        if "rank" not in row:
            issues.append(f"Row {row_num}: missing rank")
        else:
            try:
                rank = int(row["rank"])
                if rank < 1 or rank > 100:
                    issues.append(f"Row {row_num}: rank {rank} out of range")
                if rank in ranks:
                    issues.append(f"Row {row_num}: duplicate rank {rank}")
                ranks.add(rank)
            except (ValueError, TypeError):
                issues.append(f"Row {row_num}: invalid rank")
        
        if "score" not in row:
            issues.append(f"Row {row_num}: missing score")
        else:
            try:
                float(row["score"])
            except (ValueError, TypeError):
                issues.append(f"Row {row_num}: invalid score")
    
    for r in range(1, 101):
        if r not in ranks:
            issues.append(f"Missing rank: {r}")
    
    return issues


def check_scores_non_increasing(rows: List[Dict]) -> bool:
    """Verify scores are non-increasing with rank."""
    sorted_rows = sorted(rows, key=lambda x: int(x.get("rank", 0)))
    for i in range(len(sorted_rows) - 1):
        if float(sorted_rows[i]["score"]) < float(sorted_rows[i + 1]["score"]):
            return False
    return True


def is_valid_candidate_id(candidate_id: str) -> bool:
    """Check if candidate_id matches expected format."""
    import re
    return bool(re.match(r"^CAND_\d{7}$", str(candidate_id)))