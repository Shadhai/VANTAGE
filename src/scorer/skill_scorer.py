"""Skill verification and alignment scoring."""

import numpy as np
from typing import Dict, Any


class SkillScorer:
    """Verify and score candidate skills against career evidence."""
    
    def score(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Score skill claims across multiple dimensions."""
        scores = {}
        
        # 1. Skill-career alignment
        scores["alignment"] = self._score_alignment(features)
        
        # 2. Assessment performance
        scores["assessments"] = self._score_assessments(features)
        
        # 3. Skill authenticity
        scores["authenticity"] = self._score_authenticity(features)
        
        # 4. Unwanted skill penalty
        scores["unwanted_penalty"] = self._compute_unwanted_penalty(features)
        
        # 5. Suspicious pattern penalty
        scores["suspicious_penalty"] = self._compute_suspicious_penalty(features)
        
        # Combine
        scores["skill_score_total"] = self._combine_skill_scores(scores)
        
        return scores
    
    def _score_alignment(self, features: Dict) -> float:
        """Score how well claimed skills align with career evidence."""
        ai_claimed = features.get("ai_skills_claimed", 0)
        ai_verified = features.get("ai_skills_verified_in_career", 0)
        ratio = features.get("ai_skills_verified_ratio", 0)
        
        if ai_claimed == 0:
            return 0.0
        
        # Perfect alignment
        if ratio >= 0.8 and ai_claimed >= 5:
            return 1.0
        # Good alignment
        elif ratio >= 0.6:
            return 0.8
        # Moderate alignment
        elif ratio >= 0.4:
            return 0.5
        # Poor alignment - likely keyword stuffer
        elif ratio >= 0.2:
            return 0.25
        # Very poor - almost certainly stuffing
        else:
            return 0.1
    
    def _score_assessments(self, features: Dict) -> float:
        """Score based on verified platform assessments."""
        has_assessments = features.get("has_skill_assessments", False)
        count = features.get("skill_assessment_count", 0)
        avg_score = features.get("skill_assessment_avg", 0)
        max_score = features.get("skill_assessment_max", 0)
        ai_assessed = features.get("ai_skills_assessed", 0)
        
        if not has_assessments:
            return 0.3  # Neutral - no data
        
        # Weighted combination of assessment signals
        score = (
            min(1.0, count / 5) * 0.20 +
            min(1.0, avg_score / 80) * 0.30 +
            min(1.0, max_score / 90) * 0.30 +
            min(1.0, ai_assessed / 3) * 0.20
        )
        
        return score
    
    def _score_authenticity(self, features: Dict) -> float:
        """Score how genuine the skill profile appears."""
        total_skills = features.get("total_skills", 0)
        advanced = features.get("advanced_skills_count", 0)
        low_duration = features.get("advanced_skills_low_duration", 0)
        avg_duration = features.get("avg_skill_duration", 0)
        endorsements = features.get("total_endorsements", 0)
        
        if total_skills == 0:
            return 0.5
        
        # Red flags
        advanced_ratio = advanced / max(total_skills, 1)
        low_duration_ratio = low_duration / max(advanced, 1)
        
        # Too many advanced skills = suspicious
        if advanced_ratio > 0.7:
            advanced_score = 0.3
        elif advanced_ratio > 0.5:
            advanced_score = 0.5
        else:
            advanced_score = 1.0
        
        # Advanced skills with low duration = very suspicious
        if low_duration_ratio > 0.5:
            duration_score = 0.2
        elif low_duration_ratio > 0.2:
            duration_score = 0.5
        else:
            duration_score = 1.0
        
        # Endorsements per skill
        end_per_skill = endorsements / max(total_skills, 1)
        endorsement_score = min(1.0, end_per_skill / 10)
        
        return (
            advanced_score * 0.40 +
            duration_score * 0.40 +
            endorsement_score * 0.20
        )
    
    def _compute_unwanted_penalty(self, features: Dict) -> float:
        """Penalize profiles with too many irrelevant skills."""
        unwanted_ratio = features.get("unwanted_skill_ratio", 0)
        unwanted_count = features.get("unwanted_skills_count", 0)
        
        if unwanted_ratio > 0.5:
            return 0.5
        elif unwanted_ratio > 0.3:
            return 0.7
        elif unwanted_count > 5:
            return 0.85
        else:
            return 1.0
    
    def _compute_suspicious_penalty(self, features: Dict) -> float:
        """Penalize suspicious skill patterns."""
        advanced_low = features.get("advanced_skills_low_duration", 0)
        excessive = features.get("excessive_unverified_ai_skills", False)
        
        if excessive:
            return 0.3
        elif advanced_low >= 3:
            return 0.5
        elif advanced_low >= 1:
            return 0.8
        else:
            return 1.0
    
    def _combine_skill_scores(self, scores: Dict) -> float:
        """Combine all skill scores."""
        base = (
            scores["alignment"] * 0.40 +
            scores["assessments"] * 0.20 +
            scores["authenticity"] * 0.40
        )
        
        penalty = scores["unwanted_penalty"] * scores["suspicious_penalty"]
        
        return min(1.0, max(0.0, base * penalty))