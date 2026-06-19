"""Career evidence scoring using semantic matching and term analysis."""

import numpy as np
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import (
    RANKING_TERMS, ML_PRODUCTION_TERMS, EVALUATION_TERMS,
    UNWANTED_TERMS, PRODUCT_COMPANIES
)
from src.utils.text_utils import concatenate_career_text


class CareerScorer:
    """Score candidates based on career history evidence."""
    
    def __init__(self, jd_embedding: np.ndarray = None):
        self.jd_embedding = jd_embedding
    
    def score(self, features: Dict[str, Any], 
              candidate_embedding: np.ndarray = None) -> Dict[str, float]:
        """Score career evidence across multiple dimensions."""
        scores = {}
        
        # 1. Semantic similarity (if embeddings available)
        if self.jd_embedding is not None and candidate_embedding is not None:
            scores["semantic_similarity"] = self._compute_semantic_score(candidate_embedding)
        else:
            scores["semantic_similarity"] = self._compute_term_based_score(features)
        
        # 2. Ranking/retrieval evidence
        scores["ranking_evidence"] = self._score_ranking_evidence(features)
        
        # 3. ML production evidence
        scores["production_evidence"] = self._score_production_evidence(features)
        
        # 4. Evaluation experience
        scores["evaluation_evidence"] = self._score_evaluation_evidence(features)
        
        # 5. Text quality signals
        scores["text_quality"] = self._score_text_quality(features)
        
        # 6. Career progression
        scores["career_progression"] = self._score_career_progression(features)
        
        # 7. Penalties
        scores["unwanted_penalty"] = self._compute_unwanted_penalty(features)
        
        # Combine
        scores["career_evidence_total"] = self._combine_career_scores(scores, features)
        
        return scores
    
    def _compute_semantic_score(self, candidate_embedding: np.ndarray) -> float:
        """Compute cosine similarity between JD and candidate career embedding."""
        if self.jd_embedding is None or candidate_embedding is None:
            return 0.5
        
        sim = cosine_similarity(
            self.jd_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]
        
        return float(sim)
    
    def _compute_term_based_score(self, features: Dict) -> float:
        """Fallback term-based score when embeddings unavailable."""
        ranking = features.get("ranking_terms_in_career", 0)
        ml_prod = features.get("ml_production_terms_in_career", 0)
        eval_terms = features.get("evaluation_terms_in_career", 0)
        summary = features.get("summary_ranking_terms", 0) + features.get("summary_ml_terms", 0)
        
        score = (
            min(1.0, ranking / 10) * 0.40 +
            min(1.0, ml_prod / 5) * 0.30 +
            min(1.0, eval_terms / 5) * 0.20 +
            min(1.0, summary / 5) * 0.10
        )
        
        return score
    
    def _score_ranking_evidence(self, features: Dict) -> float:
        """Score based on ranking/retrieval term presence."""
        ranking_count = features.get("ranking_terms_in_career", 0)
        current_role_count = features.get("current_role_ranking_terms", 0)
        
        # Strong evidence: >5 ranking terms in career, >2 in current role
        if ranking_count >= 10 and current_role_count >= 3:
            return 1.0
        elif ranking_count >= 5 and current_role_count >= 1:
            return 0.8
        elif ranking_count >= 3:
            return 0.6
        elif ranking_count >= 1:
            return 0.3
        else:
            return 0.0
    
    def _score_production_evidence(self, features: Dict) -> float:
        """Score based on ML production deployment evidence."""
        ml_prod_count = features.get("ml_production_terms_in_career", 0)
        has_evidence = features.get("has_ml_production_evidence", False)
        
        if not has_evidence:
            return 0.0
        
        if ml_prod_count >= 5:
            return 1.0
        elif ml_prod_count >= 3:
            return 0.8
        elif ml_prod_count >= 1:
            return 0.5
        return 0.0
    
    def _score_evaluation_evidence(self, features: Dict) -> float:
        """Score based on evaluation framework experience."""
        eval_count = features.get("evaluation_terms_in_career", 0)
        has_eval = features.get("has_evaluation_evidence", False)
        
        if not has_eval:
            return 0.0
        
        if eval_count >= 5:
            return 1.0
        elif eval_count >= 3:
            return 0.7
        elif eval_count >= 1:
            return 0.4
        return 0.0
    
    def _score_text_quality(self, features: Dict) -> float:
        """Score career description quality."""
        uniqueness = features.get("career_text_uniqueness", 0.5)
        specificity = features.get("career_text_specificity", 0.5)
        avg_length = features.get("career_description_avg_length", 0)
        diversity = features.get("description_diversity", 0.5)
        
        # Longer, unique, specific descriptions = more genuine
        length_score = min(1.0, avg_length / 100)
        
        return (
            uniqueness * 0.35 +
            specificity * 0.30 +
            length_score * 0.15 +
            diversity * 0.20
        )
    
    def _score_career_progression(self, features: Dict) -> float:
        """Score based on career trajectory."""
        velocity = features.get("career_velocity", 0)
        entries = features.get("career_entries_count", 0)
        has_product = features.get("has_product_company", False)
        
        score = 0.5  # Neutral baseline
        
        if velocity > 0.3:
            score += 0.2
        if entries >= 2:
            score += 0.1
        if has_product:
            score += 0.2
        
        return min(1.0, score)
    
    def _compute_unwanted_penalty(self, features: Dict) -> float:
        """Penalize candidates with unwanted career patterns."""
        unwanted = features.get("unwanted_terms_count", 0)
        
        if unwanted >= 10:
            return 0.5
        elif unwanted >= 5:
            return 0.7
        elif unwanted >= 3:
            return 0.85
        else:
            return 1.0
    
    def _compute_jd_disqualifier_penalty(self, features: Dict) -> float:
        """Apply penalties for JD explicit disqualifiers."""
        penalty = 1.0

        # Computer vision specialist without IR/NLP
        title = str(features.get("current_title", "")).lower()
        is_cv = "computer vision" in title or "cv" in title
        has_ranking = features.get("has_ranking_evidence", False)
        has_nlp = features.get("ai_skills_claimed", 0) > 0  # Proxy

        if is_cv and not has_ranking:
            penalty *= 0.7  # 30% reduction for CV-only profiles

        # Title chaser (job hopper)
        if features.get("job_hop_risk", 0) > 0:
            penalty *= 0.85

        # Framework enthusiast (LangChain-heavy, no systems)
        # We'll use unwanted skills ratio as proxy
        if features.get("unwanted_skill_ratio", 0) > 0.4:
            penalty *= 0.9

        return penalty

    def _combine_career_scores(self, scores: Dict, features: Dict = None) -> float:
        """Combine all career evidence into final score."""
        # Primary: what they actually did
        primary = (
            scores["ranking_evidence"] * 0.30 +
            scores["production_evidence"] * 0.25 +
            scores["evaluation_evidence"] * 0.15 +
            scores["semantic_similarity"] * 0.10
        )
        
        # Secondary: quality signals
        secondary = (
            scores["text_quality"] * 0.10 +
            scores["career_progression"] * 0.10
        )
        
        combined = (primary + secondary) * scores["unwanted_penalty"]

        # Apply JD disqualifier penalties
        if features:
            combined *= self._compute_jd_disqualifier_penalty(features)
        
        return min(1.0, max(0.0, combined))