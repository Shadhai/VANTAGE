"""Coarse ranking: filter 100K → ~25K → score → top 500."""

import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from config.settings import (
    CONSULTING_COMPANIES, MIN_EXPERIENCE, MAX_EXPERIENCE,
    MAX_INACTIVE_DAYS, MIN_RESPONSE_RATE
)
from src.features.extractor import FeatureExtractor
from src.scorer.career_scorer import CareerScorer
from src.scorer.skill_scorer import SkillScorer
from src.scorer.behavioral_scorer import BehavioralScorer
from src.scorer.honeypot_detector import HoneypotDetector


class CoarseRanker:
    """First-pass ranking: filter and score all candidates."""
    
    def __init__(self, jd_embedding: np.ndarray = None):
        self.feature_extractor = FeatureExtractor()
        self.career_scorer = CareerScorer(jd_embedding)
        self.skill_scorer = SkillScorer()
        self.behavioral_scorer = BehavioralScorer()
        self.honeypot_detector = HoneypotDetector()
        self.reference_date = datetime(2026, 6, 17)
    
    def should_filter(self, candidate: Dict) -> Tuple[bool, str]:
        """Quick filter: return (should_remove, reason)."""
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        career = candidate.get("career_history", [])
        
        # 1. Experience band
        years = float(profile.get("years_of_experience", 0))
        if years < MIN_EXPERIENCE:
            return True, f"experience_too_low:{years}"
        if years > MAX_EXPERIENCE:
            # Keep only if strong ML title
            title = str(profile.get("current_title", "")).lower()
            ml_titles = ["ml engineer", "ai engineer", "data scientist", 
                        "search engineer", "ranking engineer", "machine learning"]
            if not any(t in title for t in ml_titles):
                return True, f"experience_too_high_no_ml_title:{years}"
        
        # 2. Geography
        country = str(profile.get("country", ""))
        willing = bool(signals.get("willing_to_relocate", False))
        if country.lower() != "india" and not willing:
            return True, f"non_india_no_relocation:{country}"
        
        # 3. Activity (very lenient filter - scoring handles the weight)
        last_active = str(signals.get("last_active_date", ""))
        days_inactive = self._compute_days_inactive(last_active)
        response_rate = float(signals.get("recruiter_response_rate", 0))
        
        if days_inactive > MAX_INACTIVE_DAYS and response_rate < MIN_RESPONSE_RATE:
            return True, f"ghost_profile:{days_inactive}d_inactive"
        
        # 4. Consulting-only check
        all_companies = [str(j.get("company", "")) for j in career]
        if all_companies and all(c in CONSULTING_COMPANIES for c in all_companies):
            # Check if any ML evidence despite consulting background
            career_text = " ".join([j.get("description", "") for j in career])
            ml_terms = ["ml", "machine learning", "ranking", "retrieval", "search"]
            has_ml = any(t in career_text.lower() for t in ml_terms)
            if not has_ml:
                return True, "consulting_only_no_ml"
        
        return False, "pass"
    
    def score_candidate(self, candidate: Dict, 
                        candidate_embedding: np.ndarray = None) -> Dict[str, Any]:
        """Score a single candidate across all dimensions."""
        # Extract features
        features = self.feature_extractor.extract_all(candidate)
        
        # Score each dimension
        career_scores = self.career_scorer.score(features, candidate_embedding)
        skill_scores = self.skill_scorer.score(features)
        behavioral_scores = self.behavioral_scorer.score(features)
        honeypot_result = self.honeypot_detector.detect(features)
        
        # Additional fit scores
        experience_fit = features.get("experience_band_fit", 0.5)
        location_fit = features.get("location_fit_score", 0.5)
        title_relevance = features.get("title_relevance_score", 0)
        product_fraction = features.get("product_company_fraction", 0)
        
        # Compute base score
        base_score = (
            career_scores["career_evidence_total"] * 0.40 +
            skill_scores["skill_score_total"] * 0.20 +
            experience_fit * 0.10 +
            location_fit * 0.10 +
            title_relevance * 0.10 +
            product_fraction * 0.10
        )
        
        # Apply behavioral multiplier
        behavioral_multiplier = behavioral_scores["calibrated_multiplier"]
        
        # Apply honeypot penalty
        honeypot_penalty = honeypot_result["penalty_multiplier"]
        
        # Final score
        final_score = base_score * behavioral_multiplier * honeypot_penalty
        
        return {
            "candidate_id": candidate.get("candidate_id"),
            "features": features,
            "career_scores": career_scores,
            "skill_scores": skill_scores,
            "behavioral_scores": behavioral_scores,
            "honeypot_result": honeypot_result,
            "base_score": base_score,
            "behavioral_multiplier": behavioral_multiplier,
            "honeypot_penalty": honeypot_penalty,
            "final_score": final_score,
        }
    
    def rank(self, candidates: List[Dict],
             embeddings: np.ndarray = None,
             candidate_ids: List[str] = None,
             top_k: int = 500) -> List[Dict]:
        """Rank all candidates and return top K with scores."""
        scored = []
        
        # Create ID to embedding map if available
        embedding_map = {}
        if embeddings is not None and candidate_ids is not None:
            for i, cid in enumerate(candidate_ids):
                embedding_map[cid] = embeddings[i]
        
        for candidate in candidates:
            # Quick filter
            should_remove, reason = self.should_filter(candidate)
            if should_remove:
                continue
            
            # Get embedding for this candidate
            cid = candidate.get("candidate_id", "")
            emb = embedding_map.get(cid)
            
            # Score
            result = self.score_candidate(candidate, emb)
            scored.append(result)
        
        # Sort by final score descending
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return top K
        return scored[:top_k]
    
    def _compute_days_inactive(self, last_active_str: str) -> int:
        """Compute days since last activity."""
        if not last_active_str:
            return 999
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            delta = self.reference_date - last_active
            return max(0, delta.days)
        except (ValueError, TypeError):
            return 999