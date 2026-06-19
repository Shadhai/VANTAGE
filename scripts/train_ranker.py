#!/usr/bin/env python3
"""Train XGBoost ranker on synthetic labels derived from JD interpretation.
   
   Labels are generated programmatically - no manual annotation needed.
   Model learns which features best separate JD-defined tiers.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xgboost as xgb

from config.settings import (
    CANDIDATES_FILE, PROCESSED_DIR, MODELS_DIR,
    CANDIDATE_FEATURES_FILE, JD_FILE, MODEL_FILE
)
from src.parser.jd_parser import JDParser
from src.utils.io_utils import stream_candidates, load_json


def generate_labels(candidates, jd_requirements):
    """Generate synthetic relevance labels from JD interpretation."""
    labels = []
    
    for candidate in tqdm(candidates, desc="  Labeling"):
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        signals = candidate.get("redrob_signals", {})
        
        career_text = " ".join([j.get("description", "") for j in career])
        title = str(profile.get("current_title", "")).lower()
        country = str(profile.get("country", ""))
        years = float(profile.get("years_of_experience", 0))
        
        # Tier 0: Honeypots and hard disqualifiers
        # Check salary inversion
        salary = signals.get("expected_salary_range_inr_lpa", {})
        if float(salary.get("min", 0)) > float(salary.get("max", 999)):
            labels.append(0)
            continue
        
        # Check education timeline
        education = candidate.get("education", [])
        edu_impossible = False
        for edu in education:
            start = int(edu.get("start_year", 0))
            end = int(edu.get("end_year", 0))
            if start > 0 and end > 0 and (end - start) < 2:
                edu_impossible = True
        if edu_impossible:
            labels.append(0)
            continue
        
        # Tier 1: Ideal - ranking/retrieval at product company in India
        has_ranking = any(t in career_text.lower() for t in [
            "ranking", "retrieval", "search", "recommendation", "learning-to-rank",
            "ndcg", "mrr", "embeddings", "vector search", "faiss", "pinecone"
        ])
        has_product = any(c in str(profile.get("current_company", "")) for c in [
            "Swiggy", "Zomato", "Uber", "CRED", "Razorpay", "Flipkart", "Ola",
            "InMobi", "Mad Street Den"
        ])
        in_india = country.lower() == "india"
        
        if has_ranking and has_product and in_india and 3 <= years <= 12:
            labels.append(4)  # Highest relevance
            continue
        
        # Tier 2: ML production at product company
        has_ml_prod = any(t in career_text.lower() for t in [
            "deployed", "production", "shipped", "real users", "model",
            "pipeline", "inference", "a/b test"
        ])
        has_ml_title = any(t in title for t in [
            "ml engineer", "ai engineer", "data scientist", "machine learning"
        ])
        
        if (has_ranking or has_ml_prod) and has_product and in_india:
            labels.append(3)
            continue
        
        # Tier 3: Adjacent skills with potential
        has_data_eng = any(t in career_text.lower() for t in [
            "data pipeline", "spark", "airflow", "warehouse", "feature"
        ])
        
        if has_data_eng and has_product and in_india and 3 <= years <= 10:
            labels.append(2)
            continue

        # Tier 2: Adjacent skills with transfer potential to ML/AI
        has_adjacent = any(t in career_text.lower() for t in [
            "data pipeline", "spark", "airflow", "warehouse", "feature store",
            "analytics", "ETL", "dbt", "snowflake", "bigquery", "supporting ML",
            "data science team", "feature engineering", "experimentation platform",
            "metrics", "dashboard", "BI", "reporting"
        ])

        if has_adjacent and has_product and in_india and 3 <= years <= 10:
            labels.append(2)
            continue
        
        # Tier 4: Some relevance
        if in_india and (has_ml_title or has_data_eng or has_ranking):
            labels.append(1)
            continue
        
        # Tier 5: No relevance
        labels.append(0)
    
    return np.array(labels)


def build_feature_matrix(features_df, feature_names):
    """Build feature matrix for XGBoost from parquet features."""
    X = []
    for name in feature_names:
        if name in features_df.columns:
            X.append(features_df[name].fillna(0).values)
        else:
            X.append(np.zeros(len(features_df)))
    
    return np.column_stack(X)


def main():
    print("=" * 60)
    print("Training XGBoost Ranker")
    print("=" * 60)
    
    # Load features
    print("\n[1/4] Loading pre-computed features...")
    features_df = pd.read_parquet(CANDIDATE_FEATURES_FILE)
    print(f"  Loaded {len(features_df)} candidate feature rows")
    
    # Load candidates for labeling
    print("\n[2/4] Generating synthetic labels...")
    candidates = list(stream_candidates(CANDIDATES_FILE))
    jd_requirements = load_json(PROCESSED_DIR / "jd_requirements.json")
    
    y = generate_labels(candidates, jd_requirements)
    
    # Print label distribution
    unique, counts = np.unique(y, return_counts=True)
    print("  Label distribution:")
    for label, count in zip(unique, counts):
        tier_names = {4: "Ideal", 3: "Strong ML", 2: "Adjacent", 1: "Some", 0: "None"}
        print(f"    Tier {label} ({tier_names.get(label, 'Unknown')}): {count} ({count/len(y)*100:.1f}%)")
    
    # Define feature names
    feature_names = [
        # Career evidence
        "ranking_terms_in_career", "ml_production_terms_in_career",
        "evaluation_terms_in_career", "has_ranking_evidence",
        "has_ml_production_evidence", "has_evaluation_evidence",
        "career_text_uniqueness", "career_text_specificity",
        "current_role_ranking_terms",
        
        # Skill signals
        "ai_skills_claimed", "ai_skills_verified_in_career",
        "ai_skills_verified_ratio", "skill_assessment_avg",
        "advanced_skills_low_duration", "unwanted_skills_count",
        
        # Role & company
        "title_relevance_score", "product_company_fraction",
        "experience_band_fit", "career_velocity",
        "consulting_only_flag", "job_hop_risk",
        "has_product_company",
        
        # Location
        "location_fit_score", "in_india", "in_preferred_city",
        "in_tier1_city", "willing_to_relocate",
        
        # Behavioral
        "response_rate", "interview_completion_rate",
        "days_inactive", "open_to_work", "short_notice",
        "engagement_score", "availability_score", "trust_score",
        
        # Honeypot
        "honeypot_probability", "salary_inverted",
        "excessive_unverified_ai_skills",
        
        # Text quality
        "career_description_avg_length", "description_diversity",
        "career_entries_count",
        
        # Negative signals
        "unwanted_terms_count", "unwanted_skill_ratio",
    ]
    
    # Build feature matrix
    print("\n[3/4] Building feature matrix...")
    X = build_feature_matrix(features_df, feature_names)
    print(f"  Feature matrix: {X.shape}")
    
    # Create query IDs (all same query since single JD)
    qids = np.zeros(len(y), dtype=int)
    
    # Filter to only features that exist
    valid_features = [f for f in feature_names if f in features_df.columns]
    print(f"  Using {len(valid_features)}/{len(feature_names)} features")
    
    X = build_feature_matrix(features_df, valid_features)
    
    # Train model
    print("\n[4/4] Training XGBoost ranker...")
    start_time = time.time()
    
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y, qid=qids, verbose=True)
    
    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.1f} seconds")
    
    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    print(f"✓ Model saved to {MODEL_FILE}")
    
    # Feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    print("\n  Top 10 features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, score in sorted_importance:
        feat_name = valid_features[int(feat.replace('f', ''))] if feat.startswith('f') else feat
        print(f"    {feat_name}: {score:.1f}")


if __name__ == "__main__":
    main()