"""Date parsing and recency utilities."""

from datetime import datetime, date
from typing import Optional


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string to date object."""
    if not date_str:
        return None
    
    try:
        dt = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
        return dt.date()
    except (ValueError, TypeError):
        return None


def compute_days_ago(date_str: Optional[str], reference_date: date = None) -> int:
    """Compute days between date and reference date."""
    if reference_date is None:
        reference_date = date(2026, 6, 17)
    
    parsed = parse_date(date_str)
    if parsed is None:
        return 999
    
    delta = reference_date - parsed
    return max(0, delta.days)


def is_recent(date_str: Optional[str], days: int = 30) -> bool:
    """Check if date is within recent window."""
    return compute_days_ago(date_str) <= days


def is_active_candidate(last_active: Optional[str], 
                        response_rate: float = 0,
                        open_to_work: bool = False) -> bool:
    """Determine if candidate is actively available."""
    days_inactive = compute_days_ago(last_active)
    
    if days_inactive > 180:
        return False
    if days_inactive > 90 and response_rate < 0.20:
        return False
    if not open_to_work and days_inactive > 60:
        return False
    
    return True


def format_date_for_display(date_str: Optional[str]) -> str:
    """Format date for human-readable display."""
    parsed = parse_date(date_str)
    if parsed is None:
        return "Unknown"
    return parsed.strftime("%B %Y")