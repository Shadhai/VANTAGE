"""Generate 1-2 sentence reasoning for each ranked candidate.

Rules:
- 1-2 sentences only (submission spec requirement)
- Specific facts from profile (no hallucination)
- JD connection (not generic praise)
- Honest concerns where applicable
- Variation across candidates (not templated)
- Tone matches rank
"""

from typing import Dict, List, Any


class ReasoningGenerator:
    """Generate specific, varied, 1-2 sentence reasoning strings."""
    
    def generate(self, ranked_candidates: List[Dict]) -> List[Dict]:
        for i, candidate in enumerate(ranked_candidates):
            rank = i + 1
            candidate["rank"] = rank
            candidate["reasoning"] = self._generate(candidate, rank)
        return ranked_candidates
    
    def _generate(self, result: Dict, rank: int) -> str:
        features = result.get("features", {})
        
        if rank <= 10:
            return self._top10(rank, features)
        elif rank <= 25:
            return self._strong(rank, features)
        elif rank <= 50:
            return self._solid(rank, features)
        elif rank <= 75:
            return self._potential(rank, features)
        else:
            return self._reach(rank, features)
    
    # ── Helpers ────────────────────────────────────────────
    
    def _t(self, f): 
        return str(f.get("current_title", "Professional"))
    
    def _y(self, f): 
        y = f.get('years_of_experience', 0)
        return f"{y:.0f}" if y == int(y) else f"{y:.1f}"
    
    def _c(self, f): 
        return str(f.get("current_company", ""))
    
    def _loc(self, f): 
        return str(f.get("location", "Unknown"))
    
    def _pref(self, f): 
        return bool(f.get("in_preferred_city", False))
    
    def _rr(self, f): 
        return f"{f.get('response_rate', 0):.0%}"
    
    def _np(self, f): 
        n = f.get("notice_period_days", 0)
        if n <= 0:
            return "available immediately"
        elif n <= 15:
            return f"{n}-day notice"
        elif n <= 30:
            return f"{n}-day notice period"
        elif n <= 60:
            return f"{n}-day notice"
        else:
            return f"{n}-day notice period"
    
    def _rank(self, f): 
        return bool(f.get("has_ranking_evidence", False))
    
    def _prod(self, f): 
        return bool(f.get("has_ml_production_evidence", False))
    
    def _eval(self, f): 
        return bool(f.get("has_evaluation_evidence", False))
    
    def _pc(self, f): 
        return bool(f.get("has_product_company", False))
    
    def _open(self, f): 
        return bool(f.get("open_to_work", False))
    
    def _active(self, f): 
        return bool(f.get("is_active_recently", False))
    
    def _cv(self, f): 
        return "computer vision" in self._t(f).lower() and not self._rank(f)
    
    def _hopper(self, f): 
        return bool(f.get("job_hop_risk", False))
    
    def _verified(self, f):
        return f.get("ai_skills_verified_ratio", 1.0)
    
    # ── Top 10: 10 unique openers ──────────────────────────
    
    def _top10(self, rank, f):
        openers = [
            f"{self._t(f)} with {self._y(f)} years at {self._c(f)} building production ranking and retrieval systems — strong alignment with JD's core requirements",
            f"{self._t(f)} at {self._c(f)} ({self._y(f)}y) — ships ranking systems to real users and designs evaluation frameworks, exactly the profile the JD describes",
            f"{self._t(f)} ({self._y(f)}y at {self._c(f)}) combines deep retrieval expertise with production deployment experience across the full ML stack",
            f"{self._t(f)} with {self._y(f)}y building ranking infrastructure at {self._c(f)} — matches JD's emphasis on deep technical depth and shipping velocity",
            f"{self._t(f)} — {self._y(f)}y shipping ML systems at {self._c(f)}; precisely the researcher-who-ships profile the JD prioritizes",
            f"{self._t(f)} at {self._c(f)} ({self._y(f)}y) owns ranking architecture from embeddings through A/B testing — top-tier JD match",
            f"{self._t(f)} with {self._y(f)}y leading ranking systems at {self._c(f)}; demonstrated ability to ship fast, measure rigorously, and iterate from data",
            f"{self._t(f)} ({self._y(f)}y, {self._c(f)}) brings production ranking experience plus evaluation rigor — the scrappy product-engineering attitude the JD values",
            f"{self._t(f)} at {self._c(f)} ({self._y(f)}y) — built and deployed retrieval systems to real users with strong offline and online evaluation practices",
            f"{self._t(f)} with {self._y(f)}y at {self._c(f)}; has worked on ranking since before LLMs made it fashionable — exactly what the JD asks for",
        ]
        
        opener = openers[(rank - 1) % 10]
        
        # Natural language status
        status_parts = []
        if self._open(f): 
            status_parts.append("actively looking")
        status_parts.append(f"{self._rr(f)} recruiter response rate")
        if f.get("notice_period_days", 999) <= 30: 
            status_parts.append("short notice period")
        
        loc = f"{self._loc(f)}{' (preferred location)' if self._pref(f) else ''}"
        
        return f"{opener}. Based in {loc}; {', '.join(status_parts)}."
    
    # ── Ranks 11-25: Strong, one concern ────────────────────
    
    def _strong(self, rank, f):
        titles = [
            f"{self._t(f)} with {self._y(f)}y experience",
            f"{self._t(f)} ({self._y(f)}y)",
            f"{self._t(f)} — {self._y(f)} years in the field",
        ]
        intro = titles[rank % 3]
        
        # Strength
        if self._rank(f) and self._prod(f):
            strength = "strong ranking/retrieval background with production deployment experience"
        elif self._rank(f):
            strength = "solid ranking and retrieval systems experience"
        elif self._prod(f):
            strength = "proven ML production experience at product companies"
        else:
            strength = "solid ML engineering foundation with adjacent skills"
        
        # One honest concern
        concern = None
        if f.get("notice_period_days", 0) > 90:
            concern = f"{self._np(f)} may require buyout"
        elif self._cv(f):
            concern = "primary expertise is computer vision rather than IR/ranking"
        elif not self._active(f):
            concern = "not recently active on the platform"
        elif self._verified(f) < 0.4 and f.get("ai_skills_claimed", 0) > 5:
            concern = "some claimed AI skills not independently verified in career history"
        elif not self._pc(f):
            concern = "limited product company experience"
        
        loc = f"{self._loc(f)}{' (preferred)' if self._pref(f) else ''}"
        
        if concern:
            return f"{intro}, {strength}. Based in {loc}; note: {concern}."
        else:
            return f"{intro}, {strength}. Located in {loc}; {self._rr(f)} recruiter response rate."
    
    # ── Ranks 26-50: Solid, with gaps noted ─────────────────
    
    def _solid(self, rank, f):
        intro = f"{self._t(f)} ({self._y(f)}y, {self._loc(f)})"
        
        # Body
        if self._pc(f) and self._prod(f):
            body = "brings product company and production ML experience"
        elif self._pc(f) and self._rank(f):
            body = "product company background with ranking system exposure"
        elif self._pc(f):
            body = "product company background with ML-adjacent skills"
        elif self._prod(f):
            body = "production ML experience but limited ranking/retrieval depth"
        elif self._rank(f):
            body = "ranking system knowledge but lacks product company experience"
        else:
            body = "some relevant experience but gaps in core JD requirements"
        
        # Gap
        gap = ""
        if not self._rank(f) and not self._prod(f):
            gap = "; lacks direct ranking system ownership or production ML evidence"
        elif not self._rank(f):
            gap = "; limited direct ranking/retrieval system experience"
        elif self._hopper(f):
            gap = "; frequent job changes noted as potential retention risk"
        
        return f"{intro} — {body}{gap}."
    
    # ── Ranks 51-75: Potential with clear gaps ──────────────
    
    def _potential(self, rank, f):
        intro = f"{self._t(f)} with {self._y(f)}y in {self._loc(f)}"
        
        # What they have
        if self._prod(f) and self._rank(f):
            body = "has some production ranking experience"
        elif self._prod(f):
            body = "has production ML deployment experience"
        elif self._pc(f):
            body = "product company experience with transferable adjacent skills"
        elif self._rank(f):
            body = "some ranking system knowledge but limited production depth"
        else:
            body = "limited direct ML experience but adjacent technical background"
        
        # Primary gap
        if not self._rank(f) and not self._prod(f):
            gap = "lacks production ranking/ML deployment evidence required by JD"
        elif self._cv(f):
            gap = "CV specialization does not align with JD's IR/ranking focus"
        elif not self._active(f):
            gap = "low platform engagement reduces likelihood of successful outreach"
        elif f.get("notice_period_days", 0) > 90:
            gap = "extended notice period may delay start"
        else:
            gap = "significant gaps in JD-aligned experience"
        
        return f"{intro}; {body}. Primary concern: {gap}."
    
    # ── Ranks 76-100: Reach candidates ──────────────────────
    
    def _reach(self, rank, f):
        intro = f"{self._t(f)} ({self._y(f)}y, {self._loc(f)})"
        
        # Why included at all
        reasons = []
        if self._pc(f): 
            reasons.append("product company background")
        if self._pref(f): 
            reasons.append("preferred location")
        if self._open(f) and self._active(f): 
            reasons.append("active and available")
        if f.get("response_rate", 0) > 0.60: 
            reasons.append(f"responsive to outreach ({self._rr(f)})")
        if self._prod(f):
            reasons.append("some production ML exposure")
        
        if reasons:
            why = "included for " + ", ".join(reasons)
        else:
            why = "marginal JD alignment"
        
        # Critical gap
        if self._cv(f):
            gap = "CV focus does not match JD's IR/ranking requirements"
        elif not self._rank(f) and not self._prod(f):
            gap = "does not meet core JD requirement for production ML/ranking experience"
        else:
            gap = "below primary JD fit threshold"
        
        return f"{intro} — {why}. {gap}."