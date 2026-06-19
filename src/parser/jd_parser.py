"""Parse job description into structured requirements."""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from config.settings import (
    RANKING_TERMS, ML_PRODUCTION_TERMS, EVALUATION_TERMS,
    UNWANTED_TERMS, CONSULTING_COMPANIES, PRODUCT_COMPANIES,
    PREFERRED_CITIES, TIER1_CITIES
)


class JDParser:
    """Parse and structure a job description for intelligent matching."""
    
    def __init__(self, jd_path: Path):
        self.jd_text = self._load_jd(jd_path)
        self.requirements = {}
        
    def _load_jd(self, path: Path) -> str:
        """Load job description from file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse(self) -> Dict:
        """Parse JD into structured requirements."""
        self.requirements = {
            "hard_gates": self._extract_hard_gates(),
            "strong_preferences": self._extract_strong_preferences(),
            "implicit_needs": self._extract_implicit_needs(),
            "disqualifiers": self._extract_disqualifiers(),
            "weighted_terms": self._build_weighted_terms(),
            "metadata": self._extract_metadata()
        }
        return self.requirements
    
    def _extract_hard_gates(self) -> Dict:
        """Extract binary requirements - missing = disqualified."""
        return {
            "must_have_production_ml": {
                "description": "Must have shipped ML/AI systems to production",
                "required_terms": ML_PRODUCTION_TERMS,
                "detection": self._detect_production_experience
            },
            "must_not_be_pure_research": {
                "description": "No candidates with only research background",
                "research_indicators": ["research lab", "research-only", "phd", "postdoc"],
                "requires_production_evidence": True
            },
            "must_not_be_consulting_only": {
                "description": "No candidates with only consulting background",
                "consulting_companies": CONSULTING_COMPANIES,
                "requires_product_company": True
            },
            "must_write_code": {
                "description": "Must have recent hands-on coding experience",
                "concern_titles": ["architect", "tech lead", "manager"],
                "requires_recent_coding": True
            },
            "must_be_reachable": {
                "description": "Must have reasonable response rate and recent activity",
                "min_response_rate": 0.10,
                "max_inactive_days": 180
            }
        }
    
    def _extract_strong_preferences(self) -> Dict:
        """Extract weighted preferences from JD."""
        return {
            "ranking_retrieval_experience": {
                "weight": 0.30,
                "terms": RANKING_TERMS,
                "importance": "critical"
            },
            "product_company_experience": {
                "weight": 0.20,
                "companies": PRODUCT_COMPANIES,
                "importance": "high"
            },
            "india_location": {
                "weight": 0.15,
                "preferred_cities": PREFERRED_CITIES,
                "tier1_cities": TIER1_CITIES,
                "importance": "high"
            },
            "experience_band": {
                "weight": 0.15,
                "optimal_range": [5, 9],
                "acceptable_range": [3, 12],
                "importance": "medium"
            },
            "evaluation_experience": {
                "weight": 0.10,
                "terms": EVALUATION_TERMS,
                "importance": "high"
            },
            "llm_fine_tuning": {
                "weight": 0.10,
                "terms": ["lora", "qlora", "peft", "fine-tuning", "rlhf"],
                "importance": "nice_to_have"
            }
        }
    
    def _extract_implicit_needs(self) -> Dict:
        """Extract what JD implies but doesn't state explicitly."""
        return {
            "shipper_not_researcher": {
                "weight": 0.12,
                "positive_indicators": ["shipped", "deployed", "production", 
                                       "real users", "fast", "iterate"],
                "negative_indicators": ["research paper", "novel", "state-of-the-art",
                                       "benchmark", "publication"]
            },
            "writes_well": {
                "weight": 0.08,
                "indicators": ["detailed", "specific", "documentation",
                              "design doc", "technical writing"]
            },
            "autonomous": {
                "weight": 0.05,
                "indicators": ["led", "owned", "designed", "built from scratch"]
            }
        }
    
    def _extract_disqualifiers(self) -> Dict:
        """Extract explicit disqualifying patterns."""
        return {
            "title_chaser": {
                "pattern": ">3 jobs in 5 years with title progression",
                "penalty": 0.7
            },
            "framework_enthusiast": {
                "terms": ["langchain tutorial", "langchain demo", "built a chatbot"],
                "penalty": 0.8
            },
            "cv_speech_robotics_only": {
                "terms": ["computer vision", "object detection", "yolo", 
                         "speech recognition", "tts", "robotics"],
                "condition": "primary expertise without NLP/IR",
                "penalty": 0.5
            }
        }
    
    def _build_weighted_terms(self) -> Dict[str, float]:
        """Build dictionary of all terms with weights."""
        terms = {}
        
        # Ranking terms - highest weight
        for term in RANKING_TERMS:
            terms[term.lower()] = 1.0
            
        # ML production terms
        for term in ML_PRODUCTION_TERMS:
            if term.lower() not in terms:
                terms[term.lower()] = 0.8
                
        # Evaluation terms
        for term in EVALUATION_TERMS:
            if term.lower() not in terms:
                terms[term.lower()] = 0.7
                
        # Unwanted terms - negative weight
        for term in UNWANTED_TERMS:
            terms[term.lower()] = -0.5
            
        return terms
    
    def _extract_metadata(self) -> Dict:
        """Extract role metadata from JD."""
        return {
            "title": "Senior AI Engineer",
            "company": "Redrob AI",
            "stage": "Series A",
            "team_size": "4 to 12 engineers",
            "primary_focus": "ranking, retrieval, and matching systems",
            "location_requirement": "Pune/Noida preferred, India-based",
            "experience_range": "5-9 years (flexible)",
            "culture_signals": ["async-first", "writes a lot", "disagree openly",
                               "decide quickly", "move fast"]
        }
    
    def _detect_production_experience(self, career_text: str) -> bool:
        """Check if career text indicates production ML experience."""
        combined = career_text.lower()
        return any(term in combined for term in ML_PRODUCTION_TERMS)
    
    def get_term_weights(self) -> Dict[str, float]:
        """Return weighted terms for scoring."""
        return self.requirements.get("weighted_terms", {})
    
    def get_preferences(self) -> Dict:
        """Return strong preferences."""
        return self.requirements.get("strong_preferences", {})
    
    def get_hard_gates(self) -> Dict:
        """Return hard gates."""
        return self.requirements.get("hard_gates", {})
    
    def get_implicit_needs(self) -> Dict:
        """Return implicit needs."""
        return self.requirements.get("implicit_needs", {})