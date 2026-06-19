"""Main VANTAGE pipeline orchestrator with error handling.

VANTAGE - Intelligent Candidate Ranking System
===============================================
Ranks candidates based on career evidence, not keyword matching.
Uses semantic embeddings, multi-dimensional scoring, behavioral
multipliers, and honeypot detection to produce a top-100 shortlist.

Usage:
    python src/pipeline.py --candidates data/raw/candidates.jsonl \
                           --jd data/raw/job_description.md \
                           --output output/submission.csv

Pre-requisites (run once before pipeline):
    python scripts/parse_jd.py              # Parse JD into JSON
    python scripts/precompute_embeddings.py # Generate career embeddings
    python scripts/precompute_features.py   # Extract all candidate features
    python scripts/train_ranker.py          # Train XGBoost ranker
"""

import sys
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from config.settings import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR,
    CANDIDATES_FILE, JD_FILE, EMBEDDINGS_FILE, CANDIDATE_IDS_FILE,
    CANDIDATE_FEATURES_FILE, TEXT_FREQ_FILE, MODEL_FILE, SUBMISSION_FILE,
    BEHAVIORAL_MIN, BEHAVIORAL_MAX
)
from src.parser.jd_parser import JDParser
from src.features.extractor import FeatureExtractor
from src.scorer.career_scorer import CareerScorer
from src.scorer.skill_scorer import SkillScorer
from src.scorer.behavioral_scorer import BehavioralScorer
from src.scorer.honeypot_detector import HoneypotDetector
from src.ranker.coarse_ranker import CoarseRanker
from src.ranker.fine_ranker import FineRanker
from src.ranker.tiebreaker import Tiebreaker
from src.explainer.reasoning_generator import ReasoningGenerator
from src.explainer.evidence_tracer import EvidenceTracer
from src.utils.io_utils import (
    stream_candidates, load_embeddings, load_json, save_csv
)
from src.utils.logger import PipelineLogger
from src.utils.error_handler import PipelineValidator


class VantagePipeline:
    """Complete VANTAGE ranking pipeline.
    
    Five-stage architecture:
    1. Parse JD → structured requirements
    2. Load pre-computed embeddings, features, model
    3. Coarse ranking: filter 100K → score → top 500
    4. Fine re-ranking: expand features → top 100
    5. Generate reasoning → write submission CSV
    """
    
    def __init__(self, logger: Optional[PipelineLogger] = None):
        self.logger = logger or PipelineLogger()
        self.validator = PipelineValidator()
        
        # Core components (initialized during run)
        self.jd_parser: Optional[JDParser] = None
        self.jd_embedding: Optional[np.ndarray] = None
        self.jd_requirements: Optional[Dict] = None
        
        # Pre-computed artifacts
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.candidate_ids: Optional[List[str]] = None
        self.xgboost_model: Optional[Any] = None
        self.text_frequencies: Dict[str, int] = {}
        
        # Runtime metrics
        self.error_count = 0
        self.warning_count = 0
        self.filtered_count = 0
        self.total_candidates = 0
        
        # Results
        self.top_100: List[Dict] = []
        self.submission_path: Optional[Path] = None
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def run(self,
            candidates_path: Optional[Path] = None,
            jd_path: Optional[Path] = None,
            output_path: Optional[Path] = None) -> str:
        """Run the complete VANTAGE pipeline.
        
        Args:
            candidates_path: Path to candidates.jsonl
            jd_path: Path to job_description.md
            output_path: Path for submission.csv
            
        Returns:
            str: Path to the generated submission CSV
            
        Raises:
            SystemExit: On fatal errors that prevent completion
        """
        candidates_path = candidates_path or CANDIDATES_FILE
        jd_path = jd_path or JD_FILE
        output_path = output_path or SUBMISSION_FILE
        
        self.logger.start()
        pipeline_start = time.time()
        
        try:
            # ── Stage 0: Validate Inputs ────────────────────────────
            self._stage_validate(candidates_path, jd_path)
            
            # ── Stage 1: Load Artifacts ─────────────────────────────
            self._stage_load_artifacts()
            
            # ── Stage 2: Parse Job Description ──────────────────────
            self._stage_parse_jd(jd_path)
            
            # ── Stage 3: Coarse Ranking ─────────────────────────────
            top_candidates = self._stage_coarse_ranking(candidates_path)
            
            # ── Stage 4: Fine Re-Ranking ────────────────────────────
            self.top_100 = self._stage_fine_ranking(top_candidates)
            
            # ── Stage 5: Generate Output ────────────────────────────
            self._stage_generate_output(output_path)
            
            # ── Stage 6: Validate Output ────────────────────────────
            self._stage_validate_output(output_path)
            
        except SystemExit:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected pipeline failure: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # ── Summary ─────────────────────────────────────────────────
        elapsed = time.time() - pipeline_start
        self._print_summary(elapsed)
        
        return str(output_path)
    
    # ========================================================================
    # STAGE 0: VALIDATION
    # ========================================================================
    
    def _stage_validate(self, candidates_path: Path, jd_path: Path):
        """Validate all inputs exist and are readable."""
        self.logger.section("STAGE 0: Validating Inputs")
        
        if not self.validator.validate_inputs():
            self.validator.print_issues()
            self.logger.error("Input validation failed", fatal=True)
        
        # Check file sizes
        candidates_size = candidates_path.stat().st_size / (1024**2)
        self.logger.info(f"Candidates file: {candidates_size:.1f} MB")
        self.logger.info(f"Job description: {jd_path.stat().st_size} bytes")
        
        # Quick stream test
        try:
            first = next(stream_candidates(candidates_path))
            cid = first.get("candidate_id", "UNKNOWN")
            self.logger.info(f"First candidate: {cid} - readable ✓")
        except StopIteration:
            self.logger.error("Candidates file is empty", fatal=True)
        except Exception as e:
            self.logger.error(f"Cannot read candidates file: {e}", fatal=True)
        
        # Check pre-computed status
        status = self.validator.validate_precomputed()
        missing = [k for k, v in status.items() if not v]
        if missing:
            self.logger.warning(f"Missing pre-computed artifacts: {missing}")
            self.logger.warning("Run pre-computation scripts first:")
            self.logger.warning("  python scripts/parse_jd.py")
            self.logger.warning("  python scripts/precompute_embeddings.py")
            self.logger.warning("  python scripts/precompute_features.py")
            self.logger.warning("  python scripts/train_ranker.py")
    
    # ========================================================================
    # STAGE 1: LOAD ARTIFACTS
    # ========================================================================
    
    def _stage_load_artifacts(self):
        """Load all pre-computed artifacts into memory."""
        self.logger.section("STAGE 1: Loading Pre-Computed Artifacts")
        
        # 1a. Load candidate embeddings
        self._load_embeddings()
        
        # 1b. Load candidate ID mapping
        self._load_candidate_ids()
        
        # 1c. Load text frequencies (for uniqueness scoring)
        self._load_text_frequencies()
        
        # 1d. Load XGBoost model
        self._load_xgboost_model()
        
        # 1e. Build embedding lookup map
        self._build_embedding_map()
    
    def _load_embeddings(self):
        """Load pre-computed career text embeddings."""
        if not EMBEDDINGS_FILE.exists():
            self.logger.warning(
                "No pre-computed embeddings found. "
                "Semantic matching will be limited. "
                "Run: python scripts/precompute_embeddings.py"
            )
            self.candidate_embeddings = None
            return
        
        try:
            self.candidate_embeddings = np.load(str(EMBEDDINGS_FILE))
            shape = self.candidate_embeddings.shape
            memory = self.candidate_embeddings.nbytes / (1024**2)
            self.logger.info(
                f"Loaded embeddings: {shape[0]} candidates × {shape[1]} dim "
                f"({memory:.1f} MB, {self.candidate_embeddings.dtype})"
            )
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            self.candidate_embeddings = None
    
    def _load_candidate_ids(self):
        """Load candidate ID to index mapping."""
        if not CANDIDATE_IDS_FILE.exists():
            self.logger.warning("No candidate ID mapping found.")
            self.candidate_ids = None
            return
        
        try:
            self.candidate_ids = load_json(CANDIDATE_IDS_FILE)
            self.logger.info(f"Loaded {len(self.candidate_ids)} candidate ID mappings")
            
            # Verify alignment with embeddings
            if self.candidate_embeddings is not None:
                if len(self.candidate_ids) != self.candidate_embeddings.shape[0]:
                    self.logger.warning(
                        f"ID/embedding mismatch: {len(self.candidate_ids)} IDs "
                        f"vs {self.candidate_embeddings.shape[0]} embeddings"
                    )
        except Exception as e:
            self.logger.error(f"Failed to load candidate IDs: {e}")
            self.candidate_ids = None
    
    def _load_text_frequencies(self):
        """Load career text hash frequencies for uniqueness scoring."""
        if not TEXT_FREQ_FILE.exists():
            self.logger.info("No text frequency map. Uniqueness scoring disabled.")
            return
        
        try:
            self.text_frequencies = load_json(TEXT_FREQ_FILE)
            self.logger.info(
                f"Loaded text frequency map: {len(self.text_frequencies)} unique patterns"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load text frequencies: {e}")
            self.text_frequencies = {}
    
    def _load_xgboost_model(self):
        """Load trained XGBoost ranker model."""
        if not MODEL_FILE.exists():
            self.logger.info("No XGBoost model found. Using rule-based scoring.")
            self.xgboost_model = None
            return
        
        try:
            import xgboost as xgb
            self.xgboost_model = xgb.XGBRanker()
            self.xgboost_model.load_model(str(MODEL_FILE))
            self.logger.info("Loaded XGBoost ranker model ✓")
        except Exception as e:
            self.logger.warning(f"Failed to load XGBoost model: {e}")
            self.logger.warning("Falling back to rule-based scoring.")
            self.xgboost_model = None
    
    def _build_embedding_map(self):
        """Build dictionary mapping candidate_id → embedding vector."""
        self.embedding_map = {}
        
        if self.candidate_embeddings is not None and self.candidate_ids is not None:
            for i, cid in enumerate(self.candidate_ids):
                if i < len(self.candidate_embeddings):
                    self.embedding_map[cid] = self.candidate_embeddings[i]
            
            self.logger.info(f"Built embedding lookup map: {len(self.embedding_map)} entries")
        else:
            self.logger.info("No embedding map available.")
    
    # ========================================================================
    # STAGE 2: PARSE JOB DESCRIPTION
    # ========================================================================
    
    def _stage_parse_jd(self, jd_path: Path):
        """Parse the job description into structured requirements."""
        self.logger.section("STAGE 2: Parsing Job Description")
        
        # Parse JD text into structured format
        try:
            self.jd_parser = JDParser(jd_path)
            self.jd_requirements = self.jd_parser.parse()
        except Exception as e:
            self.logger.error(f"JD parsing failed: {e}", fatal=True)
        
        # Log what we extracted
        hard_gates = self.jd_requirements.get("hard_gates", {})
        preferences = self.jd_requirements.get("strong_preferences", {})
        implicit = self.jd_requirements.get("implicit_needs", {})
        weighted_terms = self.jd_requirements.get("weighted_terms", {})
        
        self.logger.info(f"Hard gates: {len(hard_gates)} extracted")
        self.logger.info(f"Strong preferences: {len(preferences)} categories")
        self.logger.info(f"Implicit needs: {len(implicit)} identified")
        self.logger.info(f"Weighted terms: {len(weighted_terms)} terms")
        
        # Log key preferences
        for name, pref in preferences.items():
            weight = pref.get("weight", 0)
            importance = pref.get("importance", "unknown")
            self.logger.info(f"  • {name}: weight={weight:.2f}, importance={importance}")
        
        # Generate JD embedding for semantic matching
        self._generate_jd_embedding(jd_path)
    
    def _generate_jd_embedding(self, jd_path: Path):
        """Generate semantic embedding for the JD text.
        
        This is REQUIRED for proper semantic understanding.
        Without it, the system degrades to keyword matching.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            with open(jd_path, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            
            self.jd_embedding = model.encode(
                [jd_text],
                normalize_embeddings=True
            )[0].astype(np.float32)
            
            self.logger.info(
                f"JD embedding generated ✓ ({self.jd_embedding.shape[0]} dimensions)"
            )
            
        except ImportError:
            self.logger.error(
                "sentence-transformers is REQUIRED for semantic understanding.\n"
                "Install: pip install sentence-transformers\n"
                "Windows: python scripts/setup_windows.py",
                fatal=True
            )
        except OSError as e:
            self.logger.error(
                f"PyTorch/DLL error on Windows: {e}\n\n"
                "This is caused by missing Visual C++ Redistributable.\n"
                "Fix: Download and install from:\n"
                "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "Then run: python scripts/setup_windows.py",
                fatal=True
            )
        except Exception as e:
            self.logger.error(f"Failed to generate JD embedding: {e}", fatal=True)
    
    # ========================================================================
    # STAGE 3: COARSE RANKING
    # ========================================================================
    
    def _stage_coarse_ranking(self, candidates_path: Path) -> List[Dict]:
        """Stream all candidates, filter, score, and select top 500."""
        self.logger.section("STAGE 3: Coarse Ranking (100K → ~25K → 500)")
        
        # Initialize coarse ranker
        coarse_ranker = CoarseRanker(self.jd_embedding)
        
        scored = []
        self.filtered_count = 0
        self.total_candidates = 0
        self.error_count = 0
        
        filter_reasons = {}
        start_time = time.time()
        
        self.logger.info("Streaming and scoring candidates...")
        
        for candidate in stream_candidates(candidates_path):
            self.total_candidates += 1
            
            # ── Quick Filter ────────────────────────────────────
            try:
                should_remove, reason = coarse_ranker.should_filter(candidate)
            except Exception as e:
                self.error_count += 1
                if self.error_count <= 3:
                    self.logger.warning(f"Filter error at candidate {self.total_candidates}: {e}")
                continue
            
            if should_remove:
                self.filtered_count += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                continue
            
            # ── Score Candidate ─────────────────────────────────
            try:
                cid = candidate.get("candidate_id", "")
                emb = self.embedding_map.get(cid)
                result = coarse_ranker.score_candidate(candidate, emb)
                scored.append(result)
            except Exception as e:
                self.error_count += 1
                if self.error_count <= 3:
                    self.logger.warning(f"Scoring error for {cid}: {e}")
                continue
            
            # ── Progress Logging ────────────────────────────────
            if self.total_candidates % 25000 == 0:
                elapsed = time.time() - start_time
                rate = self.total_candidates / max(elapsed, 1)
                self.logger.info(
                    f"  Processed {self.total_candidates:,} candidates "
                    f"({rate:.0f}/sec) — {len(scored):,} survived, "
                    f"{self.filtered_count:,} filtered"
                )
        
        # ── Log Filtering Summary ───────────────────────────────
        elapsed = time.time() - start_time
        survival_rate = len(scored) / max(self.total_candidates, 1) * 100
        
        self.logger.info(f"Streaming complete in {elapsed:.1f}s")
        self.logger.info(f"Total candidates: {self.total_candidates:,}")
        self.logger.info(f"Survived filtering: {len(scored):,} ({survival_rate:.1f}%)")
        self.logger.info(f"Filtered out: {self.filtered_count:,}")
        self.logger.info(f"Errors: {self.error_count}")
        
        # Log top filter reasons
        if filter_reasons:
            self.logger.info("Top filter reasons:")
            sorted_reasons = sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_reasons:
                self.logger.info(f"  • {reason}: {count:,}")
        
        # ── Sort and Return Top 500 ─────────────────────────────
        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        if len(scored) < 100:
            self.logger.warning(
                f"Only {len(scored)} candidates survived. "
                "Need 100 for valid submission. Consider relaxing filters."
            )
        
        top_n = min(500, len(scored))
        self.logger.info(f"Selected top {top_n} for fine re-ranking")
        
        # Log score distribution
        if scored:
            scores = [s["final_score"] for s in scored[:top_n]]
            self.logger.info(
                f"Score range: [{min(scores):.4f}, {max(scores):.4f}], "
                f"median: {np.median(scores):.4f}"
            )
        
        return scored[:top_n]
    
    # ========================================================================
    # STAGE 4: FINE RE-RANKING
    # ========================================================================
    
    def _stage_fine_ranking(self, top_candidates: List[Dict]) -> List[Dict]:
        """Re-rank top 500 with expanded features to produce final top 100."""
        self.logger.section("STAGE 4: Fine Re-Ranking (500 → 100)")
        
        if len(top_candidates) < 100:
            self.logger.warning(
                f"Only {len(top_candidates)} candidates available for re-ranking. "
                "Output will have fewer than 100 entries."
            )
        
        # Initialize fine ranker
        fine_ranker = FineRanker(
            jd_embedding=self.jd_embedding,
            xgboost_model=self.xgboost_model
        )
        
        try:
            top_100 = fine_ranker.rerank(
                top_candidates,
                self.candidate_embeddings,
                self.candidate_ids
            )
            self.logger.info(f"Fine re-ranking complete: {len(top_100)} candidates")
        except Exception as e:
            self.logger.error(f"Fine ranking failed: {e}")
            self.logger.warning("Falling back to coarse ranking scores")
            top_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            top_100 = top_candidates[:100]
        
        # Log score distribution after re-ranking
        if top_100:
            scores = [c.get("final_score", 0) for c in top_100]
            self.logger.info(
                f"Final score range: [{min(scores):.4f}, {max(scores):.4f}]"
            )
        
        # Log diversity metrics
        titles = set()
        locations = set()
        for c in top_100:
            features = c.get("features", {})
            titles.add(str(features.get("current_title", "")))
            locations.add(str(features.get("location", "")))
        
        self.logger.info(f"Unique titles in top 100: {len(titles)}")
        self.logger.info(f"Unique locations in top 100: {len(locations)}")
        
        return top_100
    
    # ========================================================================
    # STAGE 5: GENERATE OUTPUT
    # ========================================================================
    
    def _stage_generate_output(self, output_path: Path):
        """Generate reasoning and write submission CSV."""
        self.logger.section("STAGE 5: Generating Output")
        
        # Generate reasoning for each candidate
        self._generate_reasoning()
        
        # Prepare CSV rows
        rows = self._prepare_csv_rows()
        
        # Write CSV
        self._write_submission_csv(rows, output_path)
        
        # Log top candidates
        self._log_top_candidates(rows)
    
    def _generate_reasoning(self):
        """Generate evidence-backed reasoning for each candidate."""
        self.logger.info("Generating reasoning strings...")
        
        try:
            generator = ReasoningGenerator()
            self.top_100 = generator.generate(self.top_100)
            self.logger.info(f"Generated reasoning for {len(self.top_100)} candidates ✓")
        except Exception as e:
            self.logger.error(f"Reasoning generation failed: {e}")
            self._generate_fallback_reasoning()
    
    def _generate_fallback_reasoning(self):
        """Generate minimal reasoning when the main generator fails."""
        for i, candidate in enumerate(self.top_100):
            rank = i + 1
            cid = candidate.get("candidate_id", "Unknown")
            features = candidate.get("features", {})
            title = features.get("current_title", "Professional")
            years = features.get("years_of_experience", 0)
            score = candidate.get("final_score", 0)
            
            candidate["rank"] = rank
            candidate["reasoning"] = (
                f"{title} with {years} yrs experience. "
                f"Score: {score:.4f}. "
                f"See profile for detailed qualifications."
            )
    
    def _prepare_csv_rows(self) -> List[Dict]:
        """Prepare CSV-ready rows from ranked candidates."""
        rows = []
        
        for candidate in self.top_100:
            row = {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "rank": int(candidate.get("rank", 0)),
                "score": round(float(candidate.get("final_score", 0)), 6),
                "reasoning": str(candidate.get("reasoning", "")),
            }
            rows.append(row)
        
        # Ensure sorted by rank
        rows.sort(key=lambda x: x["rank"])
        
        return rows
    
    def _write_submission_csv(self, rows: List[Dict], output_path: Path):
        """Write the final submission CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        columns = ["candidate_id", "rank", "score", "reasoning"]
        save_csv(rows, output_path, columns)
        
        size_bytes = output_path.stat().st_size
        self.logger.info(f"Submission written to: {output_path}")
        self.logger.info(f"File size: {size_bytes:,} bytes")
        self.logger.info(f"Rows: {len(rows)}")
        
        self.submission_path = output_path
    
    def _log_top_candidates(self, rows: List[Dict]):
        """Log the top-ranked candidates for quick verification."""
        self.logger.info("")
        self.logger.info("═══ TOP 10 CANDIDATES ═══")
        
        for row in rows[:10]:
            self.logger.info(
                f"  Rank {row['rank']:>3}: {row['candidate_id']} "
                f"(score: {row['score']:.4f})"
            )
            # Truncate reasoning for display
            reasoning = row['reasoning']
            if len(reasoning) > 150:
                reasoning = reasoning[:147] + "..."
            self.logger.info(f"         {reasoning}")
        
        # Also log bottom 3 to check quality gradient
        if len(rows) >= 100:
            self.logger.info(f"  ...")
            for row in rows[-3:]:
                self.logger.info(
                    f"  Rank {row['rank']:>3}: {row['candidate_id']} "
                    f"(score: {row['score']:.4f})"
                )
    
    # ========================================================================
    # STAGE 6: VALIDATE OUTPUT
    # ========================================================================
    
    def _stage_validate_output(self, output_path: Path):
        """Validate the generated submission file."""
        self.logger.section("STAGE 6: Validating Output")
        
        issues = self.validator.validate_submission_output(output_path)
        
        if not issues:
            self.logger.info("Output validation passed ✓")
            self.logger.info("  • Correct header: candidate_id, rank, score, reasoning")
            self.logger.info("  • Exactly 100 data rows")
            self.logger.info("  • All ranks 1-100 present")
            self.logger.info("  • Scores non-increasing")
            return
        
        for issue in issues:
            self.logger.error(issue)
        
        self.logger.error(
            f"Output validation failed with {len(issues)} issues. "
            "The submission may be rejected by the auto-validator.",
            fatal=True
        )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    def _print_summary(self, elapsed: float):
        """Print pipeline execution summary."""
        self.logger.section("PIPELINE SUMMARY")
        
        self.logger.info(f"Total runtime: {elapsed:.1f} seconds")
        self.logger.info(f"Candidates processed: {self.total_candidates:,}")
        self.logger.info(f"Candidates filtered: {self.filtered_count:,}")
        self.logger.info(f"Errors encountered: {self.error_count}")
        self.logger.info(f"Final output: {len(self.top_100)} candidates")
        
        if self.submission_path:
            self.logger.info(f"Submission file: {self.submission_path}")
        
        # Runtime check
        if elapsed > 300:
            self.logger.warning(
                f"Pipeline took {elapsed:.1f}s — exceeds 5-minute constraint!"
            )
        elif elapsed > 240:
            self.logger.info("Runtime within 5-minute budget (with buffer)")
        else:
            self.logger.info("Runtime well within 5-minute budget ✓")
        
        # Memory check
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            self.logger.info(f"Peak memory: {memory_mb:.0f} MB")
            if memory_mb > 14000:
                self.logger.warning("Memory usage approaching 16GB limit!")
        except ImportError:
            pass
        
        self.logger.info("")
        self.logger.info("✓ VANTAGE pipeline complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Entry point for the VANTAGE pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VANTAGE - Intelligent Candidate Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python src/pipeline.py --candidates data/raw/candidates.jsonl \\
                         --jd data/raw/job_description.md \\
                         --output output/submission.csv

  # Validate only (no ranking)
  python src/pipeline.py --validate-only

  # Custom paths
  python src/pipeline.py --candidates /path/to/candidates.jsonl \\
                         --jd /path/to/jd.md \\
                         --output /path/to/output.csv
        """
    )
    
    parser.add_argument(
        "--candidates", type=str, default=str(CANDIDATES_FILE),
        help="Path to candidates JSONL file"
    )
    parser.add_argument(
        "--jd", type=str, default=str(JD_FILE),
        help="Path to job description markdown file"
    )
    parser.add_argument(
        "--output", type=str, default=str(SUBMISSION_FILE),
        help="Path for output submission CSV"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate inputs and pre-computed artifacts, then exit"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress non-essential output"
    )
    
    args = parser.parse_args()
    
    # ── Validate-Only Mode ───────────────────────────────────────
    if args.validate_only:
        validator = PipelineValidator()
        print("=" * 60)
        print("VANTAGE - Input Validation")
        print("=" * 60)
        
        print("\nChecking inputs...")
        if not validator.validate_inputs():
            validator.print_issues()
            sys.exit(1)
        print("✓ All inputs valid")
        
        print("\nPre-computed artifacts:")
        status = validator.validate_precomputed()
        for name, exists in status.items():
            status_str = "✓ Found" if exists else "✗ Missing"
            print(f"  {name:20s}: {status_str}")
        
        sys.exit(0)
    
    # ── Full Pipeline Mode ───────────────────────────────────────
    pipeline = VantagePipeline()
    
    try:
        output_path = pipeline.run(
            candidates_path=Path(args.candidates),
            jd_path=Path(args.jd),
            output_path=Path(args.output)
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Submission ready!")
        print(f"  File: {output_path}")
        print(f"  Validate: python scripts/validate_submission.py {output_path}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user.")
        sys.exit(130)
    except SystemExit as e:
        if e.code != 0:
            print(f"\n✗ Pipeline failed. Check logs/ for details.")
        sys.exit(e.code if e.code else 1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Check logs/vantage_errors_*.log for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()