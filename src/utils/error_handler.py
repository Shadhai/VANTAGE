"""Error handling and validation for the VANTAGE pipeline."""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import (
    CANDIDATES_FILE, JD_FILE, OUTPUT_DIR, PROCESSED_DIR, MODELS_DIR
)


class PipelineValidator:
    """Validate pipeline inputs and pre-computed artifacts."""
    
    def __init__(self):
        self.issues = []
    
    def validate_inputs(self) -> bool:
        """Check all required input files exist."""
        self.issues = []
        
        # Check candidates file
        if not CANDIDATES_FILE.exists():
            self.issues.append({
                "type": "ERROR",
                "file": str(CANDIDATES_FILE),
                "message": "Candidates file not found. Place candidates.jsonl in data/raw/"
            })
        
        # Check JD file
        if not JD_FILE.exists():
            self.issues.append({
                "type": "ERROR",
                "file": str(JD_FILE),
                "message": "Job description not found. Place job_description.md in data/raw/"
            })
        
        return len([i for i in self.issues if i["type"] == "ERROR"]) == 0
    
    def validate_precomputed(self) -> Dict[str, bool]:
        """Check which pre-computed artifacts exist."""
        status = {
            "embeddings": PROCESSED_DIR / "candidate_embeddings.npy",
            "features": PROCESSED_DIR / "candidate_features.parquet",
            "jd_requirements": PROCESSED_DIR / "jd_requirements.json",
            "text_frequencies": PROCESSED_DIR / "text_frequencies.json",
            "model": MODELS_DIR / "xgboost_ranker.json",
        }
        
        return {name: path.exists() for name, path in status.items()}
    
    def validate_candidate_record(self, candidate: Dict, index: int) -> List[str]:
        """Validate a single candidate record."""
        issues = []
        
        # Check required top-level keys
        required_keys = ["candidate_id", "profile", "career_history", "skills", "redrob_signals"]
        for key in required_keys:
            if key not in candidate:
                issues.append(f"Candidate {index}: missing '{key}'")
        
        # Check candidate_id format
        cid = candidate.get("candidate_id", "")
        if not cid.startswith("CAND_") or len(cid) != 11:
            issues.append(f"Candidate {index}: invalid candidate_id '{cid}'")
        
        # Check profile
        profile = candidate.get("profile", {})
        if profile:
            if "years_of_experience" not in profile:
                issues.append(f"Candidate {index}: missing years_of_experience")
            elif not isinstance(profile["years_of_experience"], (int, float)):
                issues.append(f"Candidate {index}: years_of_experience is not numeric")
        
        # Check career_history
        career = candidate.get("career_history", [])
        if not career:
            issues.append(f"Candidate {index}: empty career_history")
        
        # Check redrob_signals
        signals = candidate.get("redrob_signals", {})
        if signals:
            if "recruiter_response_rate" not in signals:
                issues.append(f"Candidate {index}: missing recruiter_response_rate")
            if "last_active_date" not in signals:
                issues.append(f"Candidate {index}: missing last_active_date")
        
        return issues
    
    def validate_submission_output(self, output_path: Path) -> List[str]:
        """Validate the final submission CSV."""
        import csv
        issues = []
        
        if not output_path.exists():
            return ["Output file not found"]
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Check header
            try:
                header = next(reader)
            except StopIteration:
                return ["Empty output file"]
            
            expected_header = ["candidate_id", "rank", "score", "reasoning"]
            if header != expected_header:
                issues.append(f"Header mismatch: expected {expected_header}, got {header}")
            
            # Check rows
            rows = list(reader)
            if len(rows) != 100:
                issues.append(f"Expected 100 rows, got {len(rows)}")
            
            # Check ranks
            ranks = []
            ids = []
            for i, row in enumerate(rows):
                if len(row) != 4:
                    issues.append(f"Row {i+1}: expected 4 columns, got {len(row)}")
                    continue
                
                cid, rank, score, reasoning = row
                
                if not cid.startswith("CAND_"):
                    issues.append(f"Row {i+1}: invalid candidate_id '{cid}'")
                if cid in ids:
                    issues.append(f"Row {i+1}: duplicate candidate_id '{cid}'")
                ids.append(cid)
                
                try:
                    r = int(rank)
                    if r < 1 or r > 100:
                        issues.append(f"Row {i+1}: rank {r} out of range")
                    if r in ranks:
                        issues.append(f"Row {i+1}: duplicate rank {r}")
                    ranks.append(r)
                except ValueError:
                    issues.append(f"Row {i+1}: invalid rank '{rank}'")
                
                try:
                    float(score)
                except ValueError:
                    issues.append(f"Row {i+1}: invalid score '{score}'")
                
                if not reasoning or reasoning.strip() == "":
                    issues.append(f"Row {i+1}: empty reasoning")
        
        return issues
    
    def print_issues(self):
        """Print all collected issues."""
        if not self.issues:
            print("✓ No issues found")
            return
        
        errors = [i for i in self.issues if i["type"] == "ERROR"]
        warnings = [i for i in self.issues if i["type"] == "WARN"]
        
        if errors:
            print(f"\n❌ {len(errors)} Error(s):")
            for e in errors:
                print(f"  • {e['file']}: {e['message']}")
        
        if warnings:
            print(f"\n⚠️  {len(warnings)} Warning(s):")
            for w in warnings:
                print(f"  • {w['file']}: {w['message']}")


def safe_execute(func, *args, **kwargs):
    """Execute a function safely, catching and logging errors."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)