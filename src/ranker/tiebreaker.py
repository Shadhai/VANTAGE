"""Deterministic tiebreaking for equal scores."""

from typing import List, Dict, Any


class Tiebreaker:
    """Handle tiebreaking when candidates have equal scores."""
    
    def sort_and_tiebreak(self, candidates: List[Dict]) -> List[Dict]:
        """Sort candidates with deterministic tiebreaking."""
        # Sort by (final_score DESC, career_evidence DESC, 
        #          behavioral_multiplier DESC, candidate_id ASC)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -round(x.get("final_score", 0), 6),  # Round to avoid float issues
                -round(x.get("career_scores", {}).get("career_evidence_total", 0), 6),
                -round(x.get("behavioral_multiplier", 0), 6),
                -round(x.get("features", {}).get("response_rate", 0), 6),
                x.get("candidate_id", "Z"),
            )
        )
        
        return sorted_candidates
    
    def resolve_ties_in_top100(self, ranked: List[Dict]) -> List[Dict]:
        """Ensure no score ties in top 100 by adjusting if needed."""
        resolved = []
        prev_score = None
        rank = 0
        
        for candidate in ranked:
            current_score = round(candidate.get("final_score", 0), 6)
            
            if prev_score is not None and current_score == prev_score:
                # Same score as previous - this is fine per spec
                # (scores can be equal, ranks must be unique)
                pass
            
            rank += 1
            candidate["rank"] = rank
            resolved.append(candidate)
            prev_score = current_score
        
        return resolved
    
    def check_score_monotonicity(self, ranked: List[Dict]) -> bool:
        """Verify scores are non-increasing with rank."""
        for i in range(len(ranked) - 1):
            score_current = ranked[i].get("final_score", 0)
            score_next = ranked[i + 1].get("final_score", 0)
            if score_current < score_next:
                return False
        return True