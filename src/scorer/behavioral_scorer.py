"""Behavioral signal scoring and multiplier computation."""

import numpy as np
from typing import Dict, Any
from config.settings import BEHAVIORAL_MIN, BEHAVIORAL_MAX


class BehavioralScorer:
    """Score and compute behavioral multiplier from engagement signals."""
    
    def score(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Score behavioral signals and compute multiplier."""
        scores = {}
        
        # Individual dimension scores
        scores["availability"] = self._score_availability(features)
        scores["engagement"] = self._score_engagement(features)
        scores["responsiveness"] = self._score_responsiveness(features)
        scores["trust"] = self._score_trust(features)
        scores["github"] = self._score_github(features)
        scores["notice_period"] = self._score_notice_period(features)
        scores["salary"] = self._score_salary_fit(features)
        
        # Compute raw multiplier
        raw_multiplier = self._compute_raw_multiplier(scores)
        
        # Calibrate to [BEHAVIORAL_MIN, BEHAVIORAL_MAX]
        scores["raw_multiplier"] = raw_multiplier
        scores["calibrated_multiplier"] = self._calibrate(raw_multiplier)
        
        return scores
    
    def _score_availability(self, features: Dict) -> float:
        """Score how available the candidate is."""
        open_to_work = features.get("open_to_work", False)
        days_inactive = features.get("days_inactive", 999)
        is_active = features.get("is_active_recently", False)
        
        if not open_to_work:
            if days_inactive > 180:
                return 0.0
            elif days_inactive > 90:
                return 0.2
            elif days_inactive > 60:
                return 0.4
            else:
                return 0.6
        
        # Open to work
        if is_active and days_inactive <= 14:
            return 1.0
        elif is_active:
            return 0.9
        elif days_inactive <= 60:
            return 0.7
        elif days_inactive <= 90:
            return 0.5
        else:
            return 0.3
    
    def _score_engagement(self, features: Dict) -> float:
        """Score recruiter engagement signals."""
        views = features.get("profile_views_30d", 0)
        saves = features.get("saved_by_recruiters_30d", 0)
        searches = features.get("search_appearances_30d", 0)
        applications = features.get("applications_submitted_30d", 0)
        connections = features.get("connection_count", 0)
        endorsements = features.get("endorsements_received", 0)
        
        score = (
            min(1.0, views / 200) * 0.25 +
            min(1.0, saves / 20) * 0.25 +
            min(1.0, searches / 500) * 0.15 +
            min(1.0, applications / 20) * 0.10 +
            min(1.0, connections / 500) * 0.15 +
            min(1.0, endorsements / 50) * 0.10
        )
        
        return score
    
    def _score_responsiveness(self, features: Dict) -> float:
        """Score how responsive the candidate is."""
        response_rate = features.get("response_rate", 0)
        interview_rate = features.get("interview_completion_rate", 0)
        avg_response_time = features.get("avg_response_time_hours", 999)
        
        # Response rate (most important)
        if response_rate >= 0.80:
            resp_score = 1.0
        elif response_rate >= 0.60:
            resp_score = 0.8
        elif response_rate >= 0.40:
            resp_score = 0.6
        elif response_rate >= 0.20:
            resp_score = 0.4
        else:
            resp_score = 0.2
        
        # Interview completion
        if interview_rate >= 0.80:
            interview_score = 1.0
        elif interview_rate >= 0.60:
            interview_score = 0.7
        elif interview_rate >= 0.40:
            interview_score = 0.5
        else:
            interview_score = 0.3
        
        # Response time (faster = better)
        if avg_response_time < 24:
            time_score = 1.0
        elif avg_response_time < 72:
            time_score = 0.7
        elif avg_response_time < 168:  # 1 week
            time_score = 0.5
        else:
            time_score = 0.3
        
        return (
            resp_score * 0.50 +
            interview_score * 0.30 +
            time_score * 0.20
        )
    
    def _score_trust(self, features: Dict) -> float:
        """Score profile verification trust."""
        verification_count = features.get("verification_count", 0)
        fully_verified = features.get("fully_verified", False)
        completeness = features.get("profile_completeness", 0)
        linkedin = features.get("linkedin_connected", False)
        
        if fully_verified:
            base = 1.0
        elif verification_count >= 2:
            base = 0.8
        elif verification_count >= 1:
            base = 0.5
        else:
            base = 0.3
        
        # Completeness bonus
        if completeness >= 80:
            base = min(1.0, base + 0.1)
        
        # LinkedIn bonus
        if linkedin:
            base = min(1.0, base + 0.05)
        
        return base
    
    def _score_github(self, features: Dict) -> float:
        """Score GitHub activity."""
        has_github = features.get("has_github", False)
        github_score = features.get("github_activity_score", -1)
        
        if not has_github:
            return 0.3  # Neutral - no data
        
        if github_score >= 80:
            return 1.0
        elif github_score >= 50:
            return 0.8
        elif github_score >= 30:
            return 0.6
        elif github_score >= 10:
            return 0.4
        else:
            return 0.3
    
    def _score_notice_period(self, features: Dict) -> float:
        """Score notice period fit."""
        notice = features.get("notice_period_days", 999)
        short_notice = features.get("short_notice", False)
        
        # JD prefers sub-30-day notice
        if notice <= 15:
            return 1.0
        elif notice <= 30:
            return 0.9
        elif notice <= 60:
            return 0.7
        elif notice <= 90:
            return 0.5
        elif notice <= 120:
            return 0.3
        else:
            return 0.1
    
    def _score_salary_fit(self, features: Dict) -> float:
        """Score salary expectation fit (neutral unless extreme)."""
        salary_inverted = features.get("salary_inverted", False)
        salary_mid = features.get("salary_mid", 0)
        
        # Inverted salary is a honeypot signal, not a fit signal
        if salary_inverted:
            return 0.0
        
        # JD doesn't specify salary, so neutral
        # Only flag extreme outliers
        if salary_mid > 80:  # >80 LPA is very high for India
            return 0.5
        elif salary_mid < 3:  # <3 LPA is very low for senior role
            return 0.5
        
        return 0.7  # Neutral default
    
    def _compute_raw_multiplier(self, scores: Dict) -> float:
        """Compute raw behavioral multiplier from dimension scores."""
        return (
            scores["availability"] * 0.30 +
            scores["engagement"] * 0.15 +
            scores["responsiveness"] * 0.25 +
            scores["trust"] * 0.10 +
            scores["github"] * 0.05 +
            scores["notice_period"] * 0.10 +
            scores["salary"] * 0.05
        )
    
    def _calibrate(self, raw: float) -> float:
        """Calibrate multiplier to [BEHAVIORAL_MIN, BEHAVIORAL_MAX] using sigmoid."""
        # Sigmoid centered at 0.5, steepness 8
        sigmoid = 1.0 / (1.0 + np.exp(-8.0 * (raw - 0.5)))
        
        # Map to [BEHAVIORAL_MIN, BEHAVIORAL_MAX]
        calibrated = BEHAVIORAL_MIN + (BEHAVIORAL_MAX - BEHAVIORAL_MIN) * sigmoid
        
        return float(calibrated)
    
    def get_multiplier(self, features: Dict[str, Any]) -> float:
        """Convenience method: compute and return just the multiplier."""
        scores = self.score(features)
        return scores["calibrated_multiplier"]