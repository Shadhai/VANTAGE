"""Fine re-ranking: take top 500 and produce final top 100."""

import numpy as np
from typing import List, Dict, Any, Optional
from src.features.extractor import FeatureExtractor
from src.scorer.career_scorer import CareerScorer
from src.scorer.skill_scorer import SkillScorer
from src.scorer.behavioral_scorer import BehavioralScorer
from src.scorer.honeypot_detector import HoneypotDetector
from src.ranker.tiebreaker import Tiebreaker


class FineRanker:
    """Second-pass ranking with detailed feature computation."""
    
    def __init__(self, jd_embedding: np.ndarray = None, 
                 xgboost_model = None):
        self.feature_extractor = FeatureExtractor()
        self.career_scorer = CareerScorer(jd_embedding)
        self.skill_scorer = SkillScorer()
        self.behavioral_scorer = BehavioralScorer()
        self.honeypot_detector = HoneypotDetector()
        self.tiebreaker = Tiebreaker()
        self.xgboost_model = xgboost_model
    
    def rerank(self, top_candidates: List[Dict],
               embeddings: np.ndarray = None,
               candidate_ids: List[str] = None) -> List[Dict]:
        """Re-rank top candidates with expanded feature set."""
        
        # Create embedding map
        embedding_map = {}
        if embeddings is not None and candidate_ids is not None:
            for i, cid in enumerate(candidate_ids):
                embedding_map[cid] = embeddings[i]
        
        # Compute expanded scores
        for result in top_candidates:
            cid = result.get("candidate_id", "")
            features = result.get("features", {})
            
            # Compute additional fine-grained scores
            fine_scores = self._compute_fine_scores(features, result)
            result["fine_scores"] = fine_scores
            
            # If XGBoost model available, use it for final scoring
            if self.xgboost_model is not None:
                feature_vector = self._build_feature_vector(features, result)
                try:
                    xgb_score = float(self.xgboost_model.predict(
                        np.array([feature_vector])
                    )[0])
                    result["xgb_score"] = xgb_score
                    result["final_score"] = xgb_score
                except Exception:
                    # Fallback to rule-based if model fails
                    result["final_score"] = self._compute_fallback_score(result)
            else:
                result["final_score"] = self._compute_fallback_score(result)
        
        # Sort and tiebreak
        ranked = self.tiebreaker.sort_and_tiebreak(top_candidates)
        
        # Return top 100
        return ranked[:100]
    
    def _compute_fine_scores(self, features: Dict, 
                              existing_result: Dict) -> Dict[str, float]:
        """Compute additional detailed scores for fine ranking."""
        scores = {}
        
        # Transferability score (for adjacent-skill candidates)
        scores["transferability"] = self._compute_transferability(features)
        
        # Cultural fit
        scores["cultural_fit"] = features.get("cultural_fit_composite", 0.5)
        
        # Career trajectory quality
        scores["trajectory_quality"] = self._compute_trajectory_quality(features)
        
        # Skill depth vs breadth
        scores["skill_depth"] = self._compute_skill_depth(features)
        
        # Long-term potential
        scores["potential"] = self._compute_potential(features)
        
        return scores
    
    def _compute_transferability(self, features: Dict) -> float:
        """Score how transferable adjacent skills are."""
        has_ranking = features.get("has_ranking_evidence", False)
        has_ml_prod = features.get("has_ml_production_evidence", False)
        has_eval = features.get("has_evaluation_evidence", False)
        
        # Already has ML experience - no transfer needed
        if has_ranking and has_ml_prod:
            return 1.0
        
        # Adjacent signals
        signals = 0
        
        # Data engineering supporting ML
        if features.get("ml_production_terms_in_career", 0) >= 2:
            signals += 1
        
        # Product company experience
        if features.get("has_product_company", False):
            signals += 1
        
        # Career velocity (fast learner)
        if features.get("career_velocity", 0) > 0.3:
            signals += 1
        
        # GitHub activity (self-learning)
        if features.get("github_activity_score", -1) > 30:
            signals += 1
        
        # Self-awareness (honest about gaps)
        if features.get("self_awareness_signals", 0) > 0:
            signals += 1
        
        if signals >= 4:
            return 0.8
        elif signals >= 2:
            return 0.5
        else:
            return 0.2
    
    def _compute_trajectory_quality(self, features: Dict) -> float:
        """Score career trajectory quality."""
        velocity = features.get("career_velocity", 0)
        entries = features.get("career_entries_count", 0)
        product_frac = features.get("product_company_fraction", 0)
        is_hopper = features.get("job_hop_risk", 0)
        
        if is_hopper:
            return 0.2
        
        score = (
            min(1.0, velocity / 0.5) * 0.30 +
            min(1.0, entries / 3) * 0.20 +
            product_frac * 0.50
        )
        
        return score
    
    def _compute_skill_depth(self, features: Dict) -> float:
        """Score skill depth (verified expertise, not breadth)."""
        verified = features.get("ai_skills_verified_in_career", 0)
        assessments = features.get("skill_assessment_avg", 0)
        has_assessments = features.get("has_skill_assessments", False)
        
        if verified >= 5 and has_assessments and assessments > 60:
            return 1.0
        elif verified >= 3:
            return 0.7
        elif verified >= 1:
            return 0.4
        else:
            return 0.1
    
    def _compute_potential(self, features: Dict) -> float:
        """Score long-term potential."""
        velocity = features.get("career_velocity", 0)
        learning_signals = features.get("self_awareness_signals", 0)
        github = features.get("github_activity_score", -1)
        experience_fit = features.get("experience_band_fit", 0)
        
        score = (
            min(1.0, velocity / 0.5) * 0.30 +
            min(1.0, learning_signals / 3) * 0.20 +
            (min(1.0, github / 50) if github >= 0 else 0.3) * 0.15 +
            experience_fit * 0.35
        )
        
        return score
    
    def _compute_fallback_score(self, result: Dict) -> float:
        """Rule-based fallback when XGBoost unavailable."""
        career_total = result.get("career_scores", {}).get("career_evidence_total", 0)
        skill_total = result.get("skill_scores", {}).get("skill_score_total", 0)
        features = result.get("features", {})
        fine = result.get("fine_scores", {})
        
        return (
            career_total * 0.35 +
            skill_total * 0.20 +
            features.get("experience_band_fit", 0) * 0.10 +
            features.get("location_fit_score", 0) * 0.10 +
            fine.get("transferability", 0.5) * 0.10 +
            fine.get("cultural_fit", 0.5) * 0.05 +
            fine.get("trajectory_quality", 0.5) * 0.05 +
            fine.get("potential", 0.5) * 0.05
        ) * result.get("behavioral_multiplier", 1.0) * result.get("honeypot_penalty", 1.0)
    
    def _build_feature_vector(self, features: Dict, result: Dict) -> List[float]:
        """Build ordered feature vector for XGBoost prediction."""
        career = result.get("career_scores", {})
        skill = result.get("skill_scores", {})
        behavioral = result.get("behavioral_scores", {})
        fine = result.get("fine_scores", {})
        
        return [
            # Career evidence
            career.get("career_evidence_total", 0),
            career.get("ranking_evidence", 0),
            career.get("production_evidence", 0),
            career.get("evaluation_evidence", 0),
            career.get("semantic_similarity", 0),
            career.get("text_quality", 0),
            
            # Skill signals
            skill.get("skill_score_total", 0),
            skill.get("alignment", 0),
            skill.get("authenticity", 0),
            features.get("ai_skills_claimed", 0),
            features.get("ai_skills_verified_in_career", 0),
            features.get("ai_skills_verified_ratio", 0),
            features.get("skill_assessment_avg", 0),
            
            # Role & company
            features.get("title_relevance_score", 0),
            features.get("product_company_fraction", 0),
            features.get("experience_band_fit", 0),
            features.get("career_velocity", 0),
            features.get("consulting_only_flag", 0),
            features.get("job_hop_risk", 0),
            
            # Location
            features.get("location_fit_score", 0),
            features.get("in_india", 0),
            features.get("in_preferred_city", 0),
            
            # Behavioral
            behavioral.get("calibrated_multiplier", 1.0),
            behavioral.get("availability", 0),
            behavioral.get("responsiveness", 0),
            features.get("response_rate", 0),
            features.get("interview_completion_rate", 0),
            features.get("days_inactive", 999),
            features.get("open_to_work", 0),
            
            # Fine scores
            fine.get("transferability", 0),
            fine.get("cultural_fit", 0),
            fine.get("trajectory_quality", 0),
            fine.get("potential", 0),
            
            # Honeypot
            features.get("honeypot_probability", 0),
            result.get("honeypot_penalty", 1.0),
            
            # Text quality
            features.get("career_text_uniqueness", 0),
            features.get("career_text_specificity", 0),
            features.get("description_diversity", 0),
            
            # Negative signals
            features.get("unwanted_terms_count", 0),
            features.get("unwanted_skills_count", 0),
        ]