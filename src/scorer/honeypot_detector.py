"""Honeypot and trap candidate detection."""

import numpy as np
from typing import Dict, Any, Tuple


class HoneypotDetector:
    """Detect honeypot candidates and compute penalty multipliers."""
    
    def __init__(self):
        self.detection_threshold = 0.6
    
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if candidate is likely a honeypot and compute penalty."""
        result = {
            "is_honeypot": False,
            "honeypot_probability": 0.0,
            "indicators": [],
            "penalty_multiplier": 1.0,
        }
        
        # Check each indicator
        indicators = self._check_all_indicators(features)
        result["indicators"] = indicators
        
        # Count triggered indicators
        triggered = sum(1 for v in indicators.values() if v)
        
        # Compute probability
        probability = self._compute_probability(triggered, indicators)
        result["honeypot_probability"] = probability
        
        # Determine if honeypot
        result["is_honeypot"] = probability >= self.detection_threshold
        
        # Compute penalty
        result["penalty_multiplier"] = self._compute_penalty(probability)
        
        return result
    
    def _check_all_indicators(self, features: Dict) -> Dict[str, bool]:
        """Check all honeypot indicators."""
        return {
            "salary_inverted": self._check_salary_inverted(features),
            "education_impossible": self._check_education_impossible(features),
            "summary_mismatch": self._check_summary_mismatch(features),
            "excessive_unverified_skills": self._check_excessive_unverified(features),
            "completely_unverified": self._check_completely_unverified(features),
            "no_github_high_ai": self._check_no_github_high_ai(features),
            "advanced_low_duration": self._check_advanced_low_duration(features),
            "all_consulting_no_product": self._check_consulting_trap(features),
            "unwanted_skills_dominant": self._check_unwanted_dominant(features),
            "zero_engagement": self._check_zero_engagement(features),
        }
    
    def _check_salary_inverted(self, features: Dict) -> bool:
        """Check if salary min > max (data corruption)."""
        return bool(features.get("salary_inverted", False))
    
    def _check_education_impossible(self, features: Dict) -> bool:
        """Check for impossible education timelines."""
        return bool(features.get("education_timeline_impossible", False))
    
    def _check_summary_mismatch(self, features: Dict) -> bool:
        """Check if boilerplate summary doesn't match title."""
        return bool(features.get("boilerplate_summary_mismatch", False))
    
    def _check_excessive_unverified(self, features: Dict) -> bool:
        """Check for many AI skills with no career evidence."""
        ai_claimed = features.get("ai_skills_claimed", 0)
        ai_verified = features.get("ai_skills_verified_in_career", 0)
        
        if ai_claimed >= 8 and ai_verified <= 1:
            return True
        if ai_claimed >= 5 and ai_verified == 0:
            return True
        
        return False
    
    def _check_completely_unverified(self, features: Dict) -> bool:
        """Check if profile has zero verification."""
        return bool(features.get("completely_unverified", False))
    
    def _check_no_github_high_ai(self, features: Dict) -> bool:
        """Check for no GitHub but many AI skill claims."""
        has_github = features.get("has_github", False)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return not has_github and ai_claimed >= 5
    
    def _check_advanced_low_duration(self, features: Dict) -> bool:
        """Check for advanced skills with implausibly low duration."""
        advanced_low = features.get("advanced_skills_low_duration", 0)
        return advanced_low >= 3
    
    def _check_consulting_trap(self, features: Dict) -> bool:
        """Check if purely consulting background with AI keyword stuffing."""
        consulting_only = features.get("consulting_only_flag", False)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return consulting_only and ai_claimed >= 5
    
    def _check_unwanted_dominant(self, features: Dict) -> bool:
        """Check if unwanted skills dominate the profile."""
        unwanted_ratio = features.get("unwanted_skill_ratio", 0)
        unwanted_count = features.get("unwanted_skills_count", 0)
        
        return unwanted_ratio > 0.5 and unwanted_count > 5
    
    def _check_zero_engagement(self, features: Dict) -> bool:
        """Check for completely dead profiles with suspicious skill patterns."""
        days_inactive = features.get("days_inactive", 0)
        response_rate = features.get("response_rate", 0)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return days_inactive > 180 and response_rate < 0.10 and ai_claimed >= 5
    
    def _compute_probability(self, triggered: int, indicators: Dict) -> float:
        """Compute honeypot probability from triggered indicators."""
        total = len(indicators)
        if total == 0:
            return 0.0
        
        # Weighted: some indicators are stronger signals
        weights = {
            "salary_inverted": 2.0,
            "education_impossible": 2.0,
            "summary_mismatch": 1.5,
            "excessive_unverified_skills": 1.5,
            "completely_unverified": 1.0,
            "no_github_high_ai": 1.0,
            "advanced_low_duration": 1.5,
            "all_consulting_no_product": 1.0,
            "unwanted_skills_dominant": 0.5,
            "zero_engagement": 0.5,
        }
        
        weighted_sum = sum(
            weights.get(name, 1.0) for name, triggered in indicators.items() if triggered
        )
        max_weighted = sum(weights.values())
        
        return min(1.0, weighted_sum / (max_weighted * 0.6))
    
    def _compute_penalty(self, probability: float) -> float:
        """Compute penalty multiplier from honeypot probability."""
        if probability >= 0.8:
            return 0.1  # Almost certainly a honeypot
        elif probability >= 0.6:
            return 0.3  # Likely honeypot
        elif probability >= 0.4:
            return 0.5  # Suspicious
        elif probability >= 0.2:
            return 0.8  # Mildly suspicious
        else:
            return 1.0  # Clean
    
    def should_disqualify(self, features: Dict) -> bool:
        """Check if candidate should be completely disqualified."""
        detection = self.detect(features)
        return detection["honeypot_probability"] >= 0.8 
    """Honeypot and trap candidate detection."""

import numpy as np
from typing import Dict, Any, Tuple


class HoneypotDetector:
    """Detect honeypot candidates and compute penalty multipliers."""
    
    def __init__(self):
        self.detection_threshold = 0.6
    
    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if candidate is likely a honeypot and compute penalty."""
        result = {
            "is_honeypot": False,
            "honeypot_probability": 0.0,
            "indicators": [],
            "penalty_multiplier": 1.0,
        }
        
        # Check each indicator
        indicators = self._check_all_indicators(features)
        result["indicators"] = indicators
        
        # Count triggered indicators
        triggered = sum(1 for v in indicators.values() if v)
        
        # Compute probability
        probability = self._compute_probability(triggered, indicators)
        result["honeypot_probability"] = probability
        
        # Determine if honeypot
        result["is_honeypot"] = probability >= self.detection_threshold
        
        # Compute penalty
        result["penalty_multiplier"] = self._compute_penalty(probability)
        
        return result
    
    def _check_all_indicators(self, features: Dict) -> Dict[str, bool]:
        """Check all honeypot indicators."""
        return {
            "salary_inverted": self._check_salary_inverted(features),
            "education_impossible": self._check_education_impossible(features),
            "summary_mismatch": self._check_summary_mismatch(features),
            "excessive_unverified_skills": self._check_excessive_unverified(features),
            "completely_unverified": self._check_completely_unverified(features),
            "no_github_high_ai": self._check_no_github_high_ai(features),
            "advanced_low_duration": self._check_advanced_low_duration(features),
            "all_consulting_no_product": self._check_consulting_trap(features),
            "unwanted_skills_dominant": self._check_unwanted_dominant(features),
            "zero_engagement": self._check_zero_engagement(features),
        }
    
    def _check_salary_inverted(self, features: Dict) -> bool:
        """Check if salary min > max (data corruption)."""
        return bool(features.get("salary_inverted", False))
    
    def _check_education_impossible(self, features: Dict) -> bool:
        """Check for impossible education timelines."""
        return bool(features.get("education_timeline_impossible", False))
    
    def _check_summary_mismatch(self, features: Dict) -> bool:
        """Check if boilerplate summary doesn't match title."""
        return bool(features.get("boilerplate_summary_mismatch", False))
    
    def _check_excessive_unverified(self, features: Dict) -> bool:
        """Check for many AI skills with no career evidence."""
        ai_claimed = features.get("ai_skills_claimed", 0)
        ai_verified = features.get("ai_skills_verified_in_career", 0)
        
        if ai_claimed >= 8 and ai_verified <= 1:
            return True
        if ai_claimed >= 5 and ai_verified == 0:
            return True
        
        return False
    
    def _check_completely_unverified(self, features: Dict) -> bool:
        """Check if profile has zero verification."""
        return bool(features.get("completely_unverified", False))
    
    def _check_no_github_high_ai(self, features: Dict) -> bool:
        """Check for no GitHub but many AI skill claims."""
        has_github = features.get("has_github", False)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return not has_github and ai_claimed >= 5
    
    def _check_advanced_low_duration(self, features: Dict) -> bool:
        """Check for advanced skills with implausibly low duration."""
        advanced_low = features.get("advanced_skills_low_duration", 0)
        return advanced_low >= 3
    
    def _check_consulting_trap(self, features: Dict) -> bool:
        """Check if purely consulting background with AI keyword stuffing."""
        consulting_only = features.get("consulting_only_flag", False)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return consulting_only and ai_claimed >= 5
    
    def _check_unwanted_dominant(self, features: Dict) -> bool:
        """Check if unwanted skills dominate the profile."""
        unwanted_ratio = features.get("unwanted_skill_ratio", 0)
        unwanted_count = features.get("unwanted_skills_count", 0)
        
        return unwanted_ratio > 0.5 and unwanted_count > 5
    
    def _check_zero_engagement(self, features: Dict) -> bool:
        """Check for completely dead profiles with suspicious skill patterns."""
        days_inactive = features.get("days_inactive", 0)
        response_rate = features.get("response_rate", 0)
        ai_claimed = features.get("ai_skills_claimed", 0)
        
        return days_inactive > 180 and response_rate < 0.10 and ai_claimed >= 5
    
    def _compute_probability(self, triggered: int, indicators: Dict) -> float:
        """Compute honeypot probability from triggered indicators."""
        total = len(indicators)
        if total == 0:
            return 0.0
        
        # Weighted: some indicators are stronger signals
        weights = {
            "salary_inverted": 2.0,
            "education_impossible": 2.0,
            "summary_mismatch": 1.5,
            "excessive_unverified_skills": 1.5,
            "completely_unverified": 1.0,
            "no_github_high_ai": 1.0,
            "advanced_low_duration": 1.5,
            "all_consulting_no_product": 1.0,
            "unwanted_skills_dominant": 0.5,
            "zero_engagement": 0.5,
        }
        
        weighted_sum = sum(
            weights.get(name, 1.0) for name, triggered in indicators.items() if triggered
        )
        max_weighted = sum(weights.values())
        
        return min(1.0, weighted_sum / (max_weighted * 0.6))
    
    def _compute_penalty(self, probability: float) -> float:
        """Compute penalty multiplier from honeypot probability."""
        if probability >= 0.8:
            return 0.1  # Almost certainly a honeypot
        elif probability >= 0.6:
            return 0.3  # Likely honeypot
        elif probability >= 0.4:
            return 0.5  # Suspicious
        elif probability >= 0.2:
            return 0.8  # Mildly suspicious
        else:
            return 1.0  # Clean
    
    def should_disqualify(self, features: Dict) -> bool:
        """Check if candidate should be completely disqualified."""
        detection = self.detect(features)
        return detection["honeypot_probability"] >= 0.8