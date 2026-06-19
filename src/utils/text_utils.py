"""Text processing utilities."""

import re
import hashlib
from typing import List, Dict, Set
from collections import Counter


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_terms(text: str, term_list: List[str]) -> List[str]:
    """Find which terms appear in text."""
    text_lower = text.lower()
    return [term for term in term_list if term.lower() in text_lower]


def count_term_occurrences(text: str, term_list: List[str]) -> int:
    """Count how many terms appear in text."""
    text_lower = text.lower()
    return sum(1 for term in term_list if term.lower() in text_lower)


def text_contains_any(text: str, term_list: List[str]) -> bool:
    """Check if text contains any of the terms."""
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in term_list)


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from text."""
    return [float(n) for n in re.findall(r'\d+\.?\d*', text)]


def text_specificity_score(text: str) -> float:
    """Score text based on presence of specific details."""
    if not text:
        return 0.0
    
    words = text.split()
    if len(words) < 20:
        return 0.2
    
    # Count numbers and metrics
    numbers = len(re.findall(r'\d+', text))
    metrics = len(re.findall(r'\d+%|\d+x|\d+\.\d+', text))
    
    specificity = (numbers * 0.3 + metrics * 0.7) / max(len(words), 1)
    return min(1.0, specificity * 20)


def hash_text(text: str) -> str:
    """Create hash of text for frequency analysis."""
    cleaned = clean_text(text)
    return hashlib.md5(cleaned.encode()).hexdigest()


def concatenate_career_text(career_history: List[Dict]) -> str:
    """Combine all career descriptions into single text."""
    texts = []
    for job in career_history:
        desc = job.get("description", "")
        if desc:
            texts.append(desc)
    return " ".join(texts)


def concatenate_all_text(candidate: Dict) -> str:
    """Combine all text fields for analysis."""
    texts = []
    
    profile = candidate.get("profile", {})
    if profile.get("summary"):
        texts.append(profile["summary"])
    if profile.get("headline"):
        texts.append(profile["headline"])
    
    for job in candidate.get("career_history", []):
        if job.get("description"):
            texts.append(job["description"])
    
    return " ".join(texts)