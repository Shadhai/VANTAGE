"""Feature extraction from candidate profiles."""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from config.settings import (
    RANKING_TERMS, ML_PRODUCTION_TERMS, EVALUATION_TERMS,
    UNWANTED_TERMS, CONSULTING_COMPANIES, PRODUCT_COMPANIES,
    PREFERRED_CITIES, TIER1_CITIES, INDIAN_CITIES,
    CULTURAL_TERMS, EXPERIENCE_OPTIMAL, EXPERIENCE_SIGMA,
    HONEYPOT_SKILL_DURATION_MONTHS, HONEYPOT_ADVANCED_SKILLS_THRESHOLD,
    HONEYPOT_EDUCATION_MIN_YEARS
)
from src.utils.text_utils import (
    clean_text, extract_terms, count_term_occurrences,
    text_contains_any, text_specificity_score, hash_text,
    concatenate_career_text, concatenate_all_text
)


class FeatureExtractor:
    """Extract structured features from candidate profile."""
    
    def __init__(self, text_frequencies: Dict[str, int] = None):
        self.text_frequencies = text_frequencies or {}
        self.reference_date = datetime(2026, 6, 17)
    
    def extract_all(self, candidate: Dict) -> Dict[str, Any]:
        """Extract complete feature vector from candidate."""
        features = {}
        features.update(self._extract_basic_info(candidate))
        features.update(self._extract_career_evidence(candidate))
        features.update(self._extract_skill_features(candidate))
        features.update(self._extract_role_company_features(candidate))
        features.update(self._extract_location_features(candidate))
        features.update(self._extract_behavioral_features(candidate))
        features.update(self._extract_honeypot_features(candidate))
        features.update(self._extract_cultural_features(candidate))
        features.update(self._extract_composite_scores(candidate))
        return features
    
    def _extract_basic_info(self, candidate: Dict) -> Dict:
        """Extract basic candidate identifiers."""
        profile = candidate.get("profile", {})
        return {
            "candidate_id": candidate.get("candidate_id", ""),
            "years_of_experience": float(profile.get("years_of_experience", 0)),
            "current_title": str(profile.get("current_title", "")),
            "current_company": str(profile.get("current_company", "")),
            "current_industry": str(profile.get("current_industry", "")),
            "country": str(profile.get("country", "")),
            "location": str(profile.get("location", "")),
        }
    
    def _extract_career_evidence(self, candidate: Dict) -> Dict:
        """Extract ML/ranking evidence from career descriptions."""
        career_history = candidate.get("career_history", [])
        career_text = concatenate_career_text(career_history)
        all_text = concatenate_all_text(candidate)
        profile = candidate.get("profile", {})
        summary = str(profile.get("summary", ""))
        
        # Term counts in career descriptions
        ranking_count = count_term_occurrences(career_text, RANKING_TERMS)
        ml_prod_count = count_term_occurrences(career_text, ML_PRODUCTION_TERMS)
        eval_count = count_term_occurrences(career_text, EVALUATION_TERMS)
        unwanted_count = count_term_occurrences(all_text, UNWANTED_TERMS)
        
        # Also check summary for ML evidence
        summary_ranking = count_term_occurrences(summary, RANKING_TERMS)
        summary_ml = count_term_occurrences(summary, ML_PRODUCTION_TERMS)
        
        # Career text quality
        uniqueness = self._compute_uniqueness(career_text)
        specificity = text_specificity_score(career_text)
        
        # Career structure
        avg_desc_length = (
            np.mean([len(j.get("description", "").split()) for j in career_history])
            if career_history else 0
        )
        
        # Job description diversity (same text repeated = synthetic)
        descriptions = [j.get("description", "") for j in career_history]
        unique_descriptions = len(set(descriptions))
        description_diversity = unique_descriptions / max(len(descriptions), 1)
        
        # Recent role relevance (current/most recent job)
        current_job = career_history[0] if career_history else {}
        current_description = str(current_job.get("description", ""))
        current_ranking_terms = count_term_occurrences(current_description, RANKING_TERMS)
        
        return {
            "ranking_terms_in_career": ranking_count,
            "ml_production_terms_in_career": ml_prod_count,
            "evaluation_terms_in_career": eval_count,
            "unwanted_terms_count": unwanted_count,
            "summary_ranking_terms": summary_ranking,
            "summary_ml_terms": summary_ml,
            "career_text_length": len(career_text.split()),
            "career_text_uniqueness": uniqueness,
            "career_text_specificity": specificity,
            "career_description_avg_length": avg_desc_length,
            "career_entries_count": len(career_history),
            "description_diversity": description_diversity,
            "current_role_ranking_terms": current_ranking_terms,
            "has_ml_production_evidence": ml_prod_count > 0,
            "has_ranking_evidence": ranking_count > 1,
            "has_evaluation_evidence": eval_count > 0,
            "has_summary_ml_evidence": (summary_ranking + summary_ml) > 0,
        }
    
    def _extract_skill_features(self, candidate: Dict) -> Dict:
        """Extract and verify skill claims against career evidence."""
        skills = candidate.get("skills", [])
        career_text = concatenate_career_text(candidate.get("career_history", []))
        all_text = concatenate_all_text(candidate)
        
        # AI/ML skill keywords for identification
        ai_skill_patterns = RANKING_TERMS + ML_PRODUCTION_TERMS + EVALUATION_TERMS + [
            "python", "pytorch", "tensorflow", "keras", "machine learning",
            "deep learning", "nlp", "natural language", "transformer", "llm",
            "bert", "gpt", "langchain", "hugging face", "rag", "mlops",
            "kubeflow", "mlflow", "bentoml", "weights", "biases",
            "scikit-learn", "scikit", "xgboost", "lightgbm", "catboost",
            "pandas", "jupyter", "neural network", "cnn", "rnn", "lstm",
            "information retrieval", "data science", "feature engineering"
        ]
        
        # Unwanted skill patterns (indicate wrong profile)
        unwanted_skill_patterns = [
            "photoshop", "illustrator", "figma", "sketch", "adobe",
            "solidworks", "autocad", "ansys", "revit", "catia",
            "tally", "quickbooks", "sap fico", "gst",
            "content writing", "copywriting", "seo", "smm",
            "six sigma", "lean", "scrum", "pmp", "prince2",
            "salesforce", "sales cloud", "service cloud",
            "powerpoint", "excel", "word", "outlook"
        ]
        
        ai_skills_claimed = []
        ai_skills_verified = []
        unwanted_skills = []
        advanced_low_duration = 0
        total_advanced = 0
        total_endorsements = 0
        skill_durations = []
        
        for skill in skills:
            name = str(skill.get("name", "")).lower()
            proficiency = str(skill.get("proficiency", ""))
            duration = int(skill.get("duration_months", 0))
            endorsements = int(skill.get("endorsements", 0))
            
            total_endorsements += endorsements
            skill_durations.append(duration)
            
            # Identify AI skill
            if any(pattern in name for pattern in ai_skill_patterns):
                ai_skills_claimed.append(name)
                # Verify in career text
                if name in career_text.lower() or name in all_text.lower():
                    ai_skills_verified.append(name)
            
            # Identify unwanted skill
            if any(pattern in name for pattern in unwanted_skill_patterns):
                unwanted_skills.append(name)
            
            # Suspicious proficiency patterns
            if proficiency in ["advanced", "expert"]:
                total_advanced += 1
                if duration < HONEYPOT_SKILL_DURATION_MONTHS:
                    advanced_low_duration += 1
        
        # Assessment scores (verified platform assessments)
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
        assessment_values = list(assessments.values()) if assessments else []
        
        # Check if any AI skill has been assessed
        ai_skills_assessed = [s for s in ai_skills_claimed if s in assessments]
        
        return {
            "ai_skills_claimed": len(ai_skills_claimed),
            "ai_skills_verified_in_career": len(ai_skills_verified),
            "ai_skills_verified_ratio": (
                len(ai_skills_verified) / max(len(ai_skills_claimed), 1)
            ),
            "unwanted_skills_count": len(unwanted_skills),
            "unwanted_skill_ratio": len(unwanted_skills) / max(len(skills), 1),
            "total_skills": len(skills),
            "advanced_skills_count": total_advanced,
            "advanced_skills_low_duration": advanced_low_duration,
            "total_endorsements": total_endorsements,
            "avg_skill_duration": np.mean(skill_durations) if skill_durations else 0,
            "skill_assessment_count": len(assessment_values),
            "skill_assessment_avg": np.mean(assessment_values) if assessment_values else 0,
            "skill_assessment_max": max(assessment_values) if assessment_values else 0,
            "ai_skills_assessed": len(ai_skills_assessed),
            "has_skill_assessments": len(assessment_values) > 0,
        }
    
    def _extract_role_company_features(self, candidate: Dict) -> Dict:
        """Extract role fit and company quality features."""
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        
        title = str(profile.get("current_title", "")).lower()
        years = float(profile.get("years_of_experience", 0))
        
        # Title relevance scoring
        ml_titles = [
            "ml engineer", "ai engineer", "search engineer", "ranking engineer",
            "data scientist", "applied scientist", "nlp engineer", "machine learning",
            "recommendation", "relevance engineer", "retrieval engineer"
        ]
        adjacent_titles = [
            "software engineer", "backend engineer", "data engineer",
            "analytics engineer", "full stack", "platform engineer",
            "devops", "cloud engineer", "infrastructure engineer"
        ]
        non_technical_titles = [
            "marketing", "sales", "hr", "human resources", "accountant",
            "operations", "customer support", "content writer", "graphic designer",
            "mechanical engineer", "civil engineer", "electrical engineer",
            "project manager", "business analyst", "qa engineer", "tester"
        ]
        
        title_relevance = 0.0
        if any(t in title for t in ml_titles):
            title_relevance = 1.0
        elif any(t in title for t in adjacent_titles):
            title_relevance = 0.4
        elif any(t in title for t in non_technical_titles):
            title_relevance = 0.0
        
        # Company analysis
        all_companies = [str(j.get("company", "")) for j in career]
        all_industries = [str(j.get("industry", "")) for j in career]
        
        consulting_count = sum(1 for c in all_companies if c in CONSULTING_COMPANIES)
        product_count = sum(1 for c in all_companies if c in PRODUCT_COMPANIES)
        product_companies_in_career = [c for c in all_companies if c in PRODUCT_COMPANIES]
        best_company = product_companies_in_career[0] if product_companies_in_career else ""
        it_services_count = sum(1 for i in all_industries if "IT Services" in i)
        total_jobs = len(all_companies)
        
        consulting_fraction = consulting_count / max(total_jobs, 1)
        product_fraction = product_count / max(total_jobs, 1)
        it_services_fraction = it_services_count / max(total_jobs, 1)
        consulting_only = (consulting_count == total_jobs and total_jobs > 0)
        
        # Experience fit (Gaussian curve peaking at 7 years)
        experience_fit = np.exp(-((years - EXPERIENCE_OPTIMAL) ** 2) / (2 * EXPERIENCE_SIGMA ** 2))
        
        # Career progression analysis
        titles_over_time = [str(j.get("title", "")) for j in career]
        unique_titles = len(set(titles_over_time))
        career_span = years if years > 0 else 1
        velocity = unique_titles / career_span
        
        # Tenure analysis
        if total_jobs >= 2 and years > 0:
            avg_tenure = years / total_jobs
            is_job_hopper = avg_tenure < 1.5
            max_tenure = max(j.get("duration_months", 0) for j in career) / 12
        else:
            avg_tenure = years if years > 0 else 0
            is_job_hopper = False
            max_tenure = years
        
        # Current company size
        company_size_map = {
            "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
            "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8
        }
        current_size = company_size_map.get(
            str(profile.get("current_company_size", "")), 0
        )
        
        return {
            "title_relevance_score": title_relevance,
            "is_ml_title": title_relevance >= 1.0,
            "is_adjacent_title": 0.4 <= title_relevance < 1.0,
            "is_non_technical_title": title_relevance == 0.0,
            "product_company_fraction": product_fraction,
            "consulting_fraction": consulting_fraction,
            "it_services_fraction": it_services_fraction,
            "consulting_only_flag": consulting_only,
            "has_product_company": product_count > 0,
            "best_product_company": str(best_company),
            "all_product_companies": product_companies_in_career,
            "experience_band_fit": experience_fit,
            "career_velocity": min(1.0, velocity),
            "job_hop_risk": float(is_job_hopper),
            "avg_tenure_years": avg_tenure,
            "max_tenure_years": max_tenure,
            "total_jobs": total_jobs,
            "unique_companies": len(set(all_companies)),
            "current_company_size_numeric": current_size,
        }
    
    def _extract_location_features(self, candidate: Dict) -> Dict:
        """Extract location and relocation features."""
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        
        location = str(profile.get("location", ""))
        country = str(profile.get("country", ""))
        willing = bool(signals.get("willing_to_relocate", False))
        
        in_india = country.lower() == "india"
        
        # City tier detection
        in_preferred = any(c.lower() in location.lower() for c in PREFERRED_CITIES)
        in_tier1 = any(c.lower() in location.lower() for c in TIER1_CITIES)
        in_indian_city = any(c.lower() in location.lower() for c in INDIAN_CITIES)
        
        # Location fit score
        if in_preferred:
            location_fit = 1.0
        elif in_tier1:
            location_fit = 0.8
        elif in_india and in_indian_city:
            location_fit = 0.6
        elif in_india:
            location_fit = 0.4
        elif willing:
            location_fit = 0.2
        else:
            location_fit = 0.0
        
        return {
            "in_india": in_india,
            "in_preferred_city": in_preferred,
            "in_tier1_city": in_tier1,
            "in_indian_city": in_indian_city,
            "willing_to_relocate": willing,
            "location_fit_score": location_fit,
            "is_location_viable": in_india or willing,
        }
    
    def _extract_behavioral_features(self, candidate: Dict) -> Dict:
        """Extract all 23 behavioral signals."""
        signals = candidate.get("redrob_signals", {})
        
        # Recency
        last_active = str(signals.get("last_active_date", ""))
        days_inactive = self._compute_days_inactive(last_active)
        is_active_recently = days_inactive <= 30
        
        # Core response metrics
        response_rate = float(signals.get("recruiter_response_rate", 0))
        interview_rate = float(signals.get("interview_completion_rate", 0))
        offer_rate = float(signals.get("offer_acceptance_rate", -1))
        avg_response_time = float(signals.get("avg_response_time_hours", 999))
        
        # Activity metrics
        profile_views = int(signals.get("profile_views_received_30d", 0))
        search_appearances = int(signals.get("search_appearance_30d", 0))
        saved_by_recruiters = int(signals.get("saved_by_recruiters_30d", 0))
        applications = int(signals.get("applications_submitted_30d", 0))
        
        # Profile quality
        completeness = float(signals.get("profile_completeness_score", 0))
        connections = int(signals.get("connection_count", 0))
        endorsements = int(signals.get("endorsements_received", 0))
        
        # Verification
        verified_email = bool(signals.get("verified_email", False))
        verified_phone = bool(signals.get("verified_phone", False))
        linkedin = bool(signals.get("linkedin_connected", False))
        verification_count = sum([verified_email, verified_phone, linkedin])
        
        # GitHub
        github_score = float(signals.get("github_activity_score", -1))
        has_github = github_score >= 0
        
        # Availability
        open_to_work = bool(signals.get("open_to_work_flag", False))
        notice_period = int(signals.get("notice_period_days", 999))
        short_notice = notice_period <= 30
        
        # Salary
        salary = signals.get("expected_salary_range_inr_lpa", {})
        salary_min = float(salary.get("min", 0))
        salary_max = float(salary.get("max", 0))
        salary_mid = (salary_min + salary_max) / 2 if salary_max > salary_min else salary_min
        
        # Work mode
        work_mode = str(signals.get("preferred_work_mode", ""))
        
        # Composite engagement (how much recruiter interest)
        engagement = min(1.0, (
            profile_views / 200 +
            saved_by_recruiters / 20 +
            search_appearances / 500 +
            applications / 20
        ))
        
        # Composite availability (how easy to hire)
        availability = (
            (1.0 if open_to_work else 0.3) * 0.4 +
            (1.0 if is_active_recently else max(0, 1 - days_inactive/180)) * 0.3 +
            (1.0 if short_notice else max(0, 1 - notice_period/180)) * 0.2 +
            response_rate * 0.1
        )
        
        # Composite trust (how verified is this profile)
        trust = verification_count / 3.0
        
        return {
            "days_inactive": days_inactive,
            "is_active_recently": is_active_recently,
            "open_to_work": open_to_work,
            "response_rate": response_rate,
            "avg_response_time_hours": avg_response_time,
            "interview_completion_rate": interview_rate,
            "offer_acceptance_rate": offer_rate,
            "has_offer_history": offer_rate >= 0,
            "profile_completeness": completeness,
            "profile_views_30d": profile_views,
            "search_appearances_30d": search_appearances,
            "saved_by_recruiters_30d": saved_by_recruiters,
            "applications_submitted_30d": applications,
            "connection_count": connections,
            "endorsements_received": endorsements,
            "github_activity_score": github_score,
            "has_github": has_github,
            "verified_email": verified_email,
            "verified_phone": verified_phone,
            "linkedin_connected": linkedin,
            "verification_count": verification_count,
            "fully_verified": verification_count == 3,
            "notice_period_days": notice_period,
            "short_notice": short_notice,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_mid": salary_mid,
            "salary_inverted": salary_min > salary_max,
            "preferred_work_mode": work_mode,
            "engagement_score": engagement,
            "availability_score": availability,
            "trust_score": trust,
        }
    
    def _extract_honeypot_features(self, candidate: Dict) -> Dict:
        """Detect honeypot/trap patterns."""
        profile = candidate.get("profile", {})
        education = candidate.get("education", [])
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        signals = candidate.get("redrob_signals", {})
        
        # 1. Salary inversion
        salary = signals.get("expected_salary_range_inr_lpa", {})
        salary_inverted = float(salary.get("min", 0)) > float(salary.get("max", 999))
        
        # 2. Education timeline impossible
        edu_impossible = False
        for edu in education:
            start = int(edu.get("start_year", 0))
            end = int(edu.get("end_year", 0))
            if start > 0 and end > 0:
                duration = end - start
                if duration < HONEYPOT_EDUCATION_MIN_YEARS:
                    edu_impossible = True
                    break
        
        # 3. Boilerplate summary mismatch
        summary = str(profile.get("summary", "")).lower()
        title = str(profile.get("current_title", "")).lower()
        is_marketing_summary = "marketing manager" in summary
        is_unrelated_title = not any(t in title for t in ["marketing", "sales", "business"])
        summary_mismatch = is_marketing_summary and is_unrelated_title
        
        # 4. Excessive unverified AI skills
        career_text = concatenate_career_text(career)
        ai_skill_count = 0
        ai_verified = 0
        for skill in skills:
            name = str(skill.get("name", "")).lower()
            proficiency = str(skill.get("proficiency", ""))
            if proficiency in ["advanced", "expert"]:
                if any(term in name for term in RANKING_TERMS + ["nlp", "llm", "transformer", "pytorch", "tensorflow"]):
                    ai_skill_count += 1
                    if name in career_text:
                        ai_verified += 1
        
        excessive_unverified = (
            ai_skill_count >= HONEYPOT_ADVANCED_SKILLS_THRESHOLD and 
            ai_verified <= 1
        )
        
        # 5. Completely unverified profile
        verified_email = bool(signals.get("verified_email", False))
        verified_phone = bool(signals.get("verified_phone", False))
        linkedin = bool(signals.get("linkedin_connected", False))
        completely_unverified = not (verified_email or verified_phone or linkedin)
        
        # 6. GitHub missing with high AI claims
        github_score = float(signals.get("github_activity_score", -1))
        no_github_high_ai = (github_score < 0 and ai_skill_count >= HONEYPOT_ADVANCED_SKILLS_THRESHOLD)
        
        # 7. Description template duplication (marker)
        descriptions = [j.get("description", "") for j in career]
        unique_descriptions = len(set(descriptions))
        all_templated = unique_descriptions < len(descriptions) and len(descriptions) >= 2
        
        # Combine honeypot probability
        honeypot_indicators = sum([
            salary_inverted,
            edu_impossible,
            summary_mismatch,
            excessive_unverified,
            completely_unverified,
            no_github_high_ai,
            all_templated and excessive_unverified,
        ])
        
        honeypot_probability = min(1.0, honeypot_indicators / 3.0)
        
        return {
            "salary_inverted": salary_inverted,
            "education_timeline_impossible": edu_impossible,
            "boilerplate_summary_mismatch": summary_mismatch,
            "excessive_unverified_ai_skills": excessive_unverified,
            "completely_unverified": completely_unverified,
            "no_github_high_ai_skills": no_github_high_ai,
            "all_descriptions_templated": all_templated,
            "honeypot_indicator_count": honeypot_indicators,
            "honeypot_probability": honeypot_probability,
            "is_likely_honeypot": honeypot_probability >= 0.6,
        }
    
    def _extract_cultural_features(self, candidate: Dict) -> Dict:
        """Extract cultural fit signals from text."""
        all_text = concatenate_all_text(candidate)
        profile = candidate.get("profile", {})
        summary = str(profile.get("summary", ""))
        
        # Decisiveness
        decisiveness = count_term_occurrences(all_text, CULTURAL_TERMS.get("decisiveness", []))
        
        # Bias for action
        action_bias = count_term_occurrences(all_text, CULTURAL_TERMS.get("bias_for_action", []))
        
        # Autonomy
        autonomy = count_term_occurrences(all_text, CULTURAL_TERMS.get("autonomy", []))
        
        # Self-awareness (honesty about limitations)
        self_awareness = count_term_occurrences(all_text, CULTURAL_TERMS.get("self_awareness", []))
        
        # Writing quality
        specificity = text_specificity_score(all_text)
        has_metrics = len([w for w in all_text.split() if any(c.isdigit() for c in w)]) > 5
        
        # Shipper vs researcher
        research_terms = ["research paper", "novel", "state-of-the-art", "benchmark", "publication"]
        shipper_terms = ["shipped", "deployed", "production", "real users", "launched"]
        
        research_score = count_term_occurrences(all_text, research_terms)
        shipper_score = count_term_occurrences(all_text, shipper_terms)
        
        shipper_ratio = shipper_score / max(research_score + shipper_score, 1)
        
        return {
            "decisiveness_signals": decisiveness,
            "action_bias_signals": action_bias,
            "autonomy_signals": autonomy,
            "self_awareness_signals": self_awareness,
            "writing_specificity": specificity,
            "has_metrics_in_text": has_metrics,
            "shipper_score": shipper_score,
            "researcher_score": research_score,
            "shipper_ratio": shipper_ratio,
            "cultural_fit_composite": np.mean([
                min(1.0, decisiveness / 5),
                min(1.0, action_bias / 5),
                min(1.0, autonomy / 5),
                specificity,
                shipper_ratio,
            ]),
        }
    
    def _extract_composite_scores(self, candidate: Dict) -> Dict:
        """Compute composite scores that combine multiple signals."""
        # Will be enriched after all features extracted
        return {
            "composite_career_strength": 0.0,  # Computed during scoring
            "composite_skill_authenticity": 0.0,
            "composite_hireability": 0.0,
        }
    
    def _compute_uniqueness(self, text: str) -> float:
        """Score text uniqueness based on frequency in dataset."""
        if not text or not self.text_frequencies:
            return 0.5  # Unknown
        
        text_hash = hash_text(text)
        frequency = self.text_frequencies.get(text_hash, 1)
        
        if frequency > 100:
            return 0.3
        elif frequency > 20:
            return 0.6
        elif frequency > 5:
            return 0.8
        else:
            return 1.0
    
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
    
    def _compute_location_score(self, in_india: bool, in_preferred: bool, 
                                  in_tier1: bool, willing: bool) -> float:
        """Compute location fit score."""
        if in_preferred:
            return 1.0
        elif in_tier1:
            return 0.8
        elif in_india:
            return 0.5
        elif willing:
            return 0.3
        else:
            return 0.0
    
    def _compute_engagement(self, views: int, searches: int, 
                            saves: int, applications: int) -> float:
        """Compute recruiter engagement score."""
        return min(1.0, (
            views / 200 + 
            searches / 500 + 
            saves / 20 + 
            applications / 20
        ))
    
    def _compute_availability(self, open_to_work: bool, days_inactive: int,
                               notice_period: int, response_rate: float) -> float:
        """Compute candidate availability score."""
        score = 0.0
        score += (1.0 if open_to_work else 0.3) * 0.4
        score += max(0, 1.0 - days_inactive / 180) * 0.3
        score += max(0, 1.0 - notice_period / 180) * 0.2
        score += response_rate * 0.1
        return score
    
    def _compute_trust(self, email: bool, phone: bool, linkedin: bool) -> float:
        """Compute profile trustworthiness score."""
        return sum([email, phone, linkedin]) / 3.0