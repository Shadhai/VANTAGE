"""Trace scores back to specific evidence for explainability."""

from typing import Dict, List, Any, Tuple
from config.settings import PREFERRED_CITIES, TIER1_CITIES


class EvidenceTracer:
    """Trace every score component to its source evidence."""
    
    def trace(self, result: Dict) -> Dict[str, Any]:
        """Generate full evidence trace for a candidate."""
        features = result.get("features", {})
        career_scores = result.get("career_scores", {})
        skill_scores = result.get("skill_scores", {})
        behavioral_scores = result.get("behavioral_scores", {})
        honeypot = result.get("honeypot_result", {})
        fine_scores = result.get("fine_scores", {})
        
        trace = {
            "candidate_id": result.get("candidate_id"),
            "final_score": result.get("final_score", 0),
            "score_breakdown": {
                "career_evidence": self._trace_career(features, career_scores),
                "skills": self._trace_skills(features, skill_scores),
                "role_fit": self._trace_role(features),
                "location": self._trace_location(features),
                "behavioral": self._trace_behavioral(features, behavioral_scores),
                "transferability": self._trace_transferability(features, fine_scores),
            },
            "penalties_applied": self._trace_penalties(result),
            "key_strengths": self._extract_strengths(features, career_scores),
            "key_gaps": self._extract_gaps(features, career_scores),
        }
        
        return trace
    
    def _trace_career(self, features: Dict, scores: Dict) -> Dict:
        """Trace career evidence score."""
        return {
            "total": scores.get("career_evidence_total", 0),
            "ranking_terms_found": features.get("ranking_terms_in_career", 0),
            "ml_production_terms_found": features.get("ml_production_terms_in_career", 0),
            "evaluation_terms_found": features.get("evaluation_terms_in_career", 0),
            "has_ranking_evidence": features.get("has_ranking_evidence", False),
            "has_production_evidence": features.get("has_ml_production_evidence", False),
            "has_evaluation_evidence": features.get("has_evaluation_evidence", False),
            "career_text_uniqueness": features.get("career_text_uniqueness", 0),
            "career_text_specificity": features.get("career_text_specificity", 0),
        }
    
    def _trace_skills(self, features: Dict, scores: Dict) -> Dict:
        """Trace skill verification."""
        return {
            "total": scores.get("skill_score_total", 0),
            "ai_skills_claimed": features.get("ai_skills_claimed", 0),
            "ai_skills_verified": features.get("ai_skills_verified_in_career", 0),
            "verification_ratio": features.get("ai_skills_verified_ratio", 0),
            "assessment_avg": features.get("skill_assessment_avg", 0),
            "suspicious_patterns": features.get("advanced_skills_low_duration", 0),
        }
    
    def _trace_role(self, features: Dict) -> Dict:
        """Trace role and company fit."""
        return {
            "current_title": features.get("current_title", ""),
            "title_relevance": features.get("title_relevance_score", 0),
            "years_of_experience": features.get("years_of_experience", 0),
            "experience_band_fit": features.get("experience_band_fit", 0),
            "product_company_fraction": features.get("product_company_fraction", 0),
            "has_product_company": features.get("has_product_company", False),
            "consulting_only": features.get("consulting_only_flag", False),
            "career_velocity": features.get("career_velocity", 0),
        }
    
    def _trace_location(self, features: Dict) -> Dict:
        """Trace location fit."""
        return {
            "location": features.get("location", ""),
            "country": features.get("country", ""),
            "in_india": features.get("in_india", False),
            "in_preferred_city": features.get("in_preferred_city", False),
            "in_tier1_city": features.get("in_tier1_city", False),
            "willing_to_relocate": features.get("willing_to_relocate", False),
            "location_fit_score": features.get("location_fit_score", 0),
        }
    
    def _trace_behavioral(self, features: Dict, scores: Dict) -> Dict:
        """Trace behavioral signals."""
        return {
            "multiplier": scores.get("calibrated_multiplier", 1.0),
            "response_rate": features.get("response_rate", 0),
            "interview_completion": features.get("interview_completion_rate", 0),
            "days_inactive": features.get("days_inactive", 0),
            "open_to_work": features.get("open_to_work", False),
            "notice_period_days": features.get("notice_period_days", 0),
            "availability_score": scores.get("availability", 0),
            "responsiveness_score": scores.get("responsiveness", 0),
        }
    
    def _trace_transferability(self, features: Dict, fine_scores: Dict) -> Dict:
        """Trace transferability assessment."""
        return {
            "score": fine_scores.get("transferability", 0),
            "has_direct_ml_experience": features.get("has_ranking_evidence", False),
            "career_velocity": features.get("career_velocity", 0),
            "self_awareness": features.get("self_awareness_signals", 0),
        }
    
    def _trace_penalties(self, result: Dict) -> Dict:
        """Trace all penalties applied."""
        return {
            "honeypot_penalty": result.get("honeypot_penalty", 1.0),
            "honeypot_probability": result.get("honeypot_result", {}).get("honeypot_probability", 0),
            "is_likely_honeypot": result.get("honeypot_result", {}).get("is_honeypot", False),
            "honeypot_indicators": result.get("honeypot_result", {}).get("indicators", []),
            "behavioral_multiplier": result.get("behavioral_multiplier", 1.0),
        }
    
    def _extract_strengths(self, features: Dict, scores: Dict) -> List[str]:
        """Extract key strengths for explanation."""
        strengths = []
        
        if features.get("has_ranking_evidence"):
            strengths.append("ranking_retrieval_experience")
        if features.get("has_ml_production_evidence"):
            strengths.append("ml_production_experience")
        if features.get("has_evaluation_evidence"):
            strengths.append("evaluation_experience")
        if features.get("has_product_company"):
            strengths.append("product_company_experience")
        if features.get("in_preferred_city"):
            strengths.append("preferred_location")
        if features.get("in_tier1_city"):
            strengths.append("tier1_location")
        if features.get("response_rate", 0) > 0.70:
            strengths.append("high_response_rate")
        if features.get("open_to_work"):
            strengths.append("open_to_work")
        if features.get("career_velocity", 0) > 0.3:
            strengths.append("career_growth")
        if features.get("ai_skills_verified_ratio", 0) > 0.7:
            strengths.append("verified_skills")
        
        return strengths
    
    def _extract_gaps(self, features: Dict, scores: Dict) -> List[str]:
        """Extract key gaps for explanation."""
        gaps = []
        
        if not features.get("has_ranking_evidence"):
            gaps.append("no_ranking_experience")
        if not features.get("has_ml_production_evidence"):
            gaps.append("no_ml_production")
        if not features.get("has_product_company"):
            gaps.append("no_product_company")
        if features.get("consulting_only_flag"):
            gaps.append("consulting_only_background")
        if features.get("days_inactive", 0) > 90:
            gaps.append("inactive_profile")
        if features.get("response_rate", 0) < 0.30:
            gaps.append("low_response_rate")
        if not features.get("in_india") and not features.get("willing_to_relocate"):
            gaps.append("location_mismatch")
        if features.get("ai_skills_verified_ratio", 0) < 0.3 and features.get("ai_skills_claimed", 0) > 5:
            gaps.append("unverified_skills")
        if features.get("notice_period_days", 0) > 90:
            gaps.append("long_notice_period")
        
        return gaps