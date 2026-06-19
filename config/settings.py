"""Global configuration for VANTAGE."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"
OUTPUT_DIR = BASE_DIR / "output"

# Files
CANDIDATES_FILE = RAW_DIR / "candidates.jsonl"
JD_FILE = RAW_DIR / "job_description.md"
CANDIDATE_FEATURES_FILE = PROCESSED_DIR / "candidate_features.parquet"
EMBEDDINGS_FILE = PROCESSED_DIR / "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = PROCESSED_DIR / "candidate_ids.json"
TEXT_FREQ_FILE = PROCESSED_DIR / "text_frequencies.json"
MODEL_FILE = MODELS_DIR / "xgboost_ranker.json"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# Runtime constraints
MAX_RUNTIME_SECONDS = 300  # 5 minutes
MAX_MEMORY_GB = 16
CPU_ONLY = True

# Scoring weights (initial, will be learned)
SIGNAL_WEIGHTS = {
    "career_evidence": 0.40,
    "skill_alignment": 0.25,
    "role_company_fit": 0.15,
    "jd_requirement_match": 0.20
}

# Behavioral multiplier bounds
BEHAVIORAL_MIN = 0.4
BEHAVIORAL_MAX = 1.3

# Experience curve
EXPERIENCE_OPTIMAL = 7.0
EXPERIENCE_SIGMA = 3.0

# Filter thresholds
MIN_EXPERIENCE = 2
MAX_EXPERIENCE = 14
MAX_INACTIVE_DAYS = 180
MIN_RESPONSE_RATE = 0.05

# Honeypot thresholds
HONEYPOT_SALARY_INVERTED = True
HONEYPOT_SKILL_DURATION_MONTHS = 6
HONEYPOT_ADVANCED_SKILLS_THRESHOLD = 5
HONEYPOT_EDUCATION_MIN_YEARS = 2

# Preferred locations (in order)
PREFERRED_CITIES = ["Pune", "Noida"]
TIER1_CITIES = ["Mumbai", "Delhi", "Hyderabad", "Bangalore", "Gurgaon", "Chennai"]
INDIAN_CITIES = TIER1_CITIES + ["Kochi", "Kolkata", "Chandigarh", "Trivandrum", 
                                 "Coimbatore", "Indore", "Bhubaneswar", "Ahmedabad",
                                 "Vizag", "Lucknow", "Jaipur", "Nagpur"]

# Consulting companies
CONSULTING_COMPANIES = ["TCS", "Infosys", "Wipro", "Cognizant", "HCL", 
                        "Tech Mahindra", "Mindtree", "Mphasis", "L&T Infotech"]

# Product companies (known from dataset)
PRODUCT_COMPANIES = ["Swiggy", "Zomato", "Uber", "CRED", "Razorpay", "Flipkart",
                     "Ola", "InMobi", "Mad Street Den", "Pied Piper", "Hooli",
                     "Initech", "Globex Inc", "Stark Industries", "Wayne Enterprises",
                     "Acme Corp", "Dunder Mifflin", "Redrob"]

# JD requirements - ranking/retrieval terms
RANKING_TERMS = [
    "ranking", "retrieval", "search", "recommendation", "embeddings",
    "vector", "semantic", "learning-to-rank", "xgboost", "lightgbm",
    "ndcg", "mrr", "map", "a/b test", "offline evaluation",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "sentence-transformers", "bge", "e5", "hybrid search",
    "fine-tuning", "lora", "qlora", "peft", "bm25", "relevance",
    "query", "candidate", "matching", "scoring", "rerank"
]

# ML production terms
ML_PRODUCTION_TERMS = [
    "deployed", "production", "shipped", "real users", "scale",
    "pipeline", "serving", "inference", "monitoring", "a/b"
]

# Evaluation terms
EVALUATION_TERMS = [
    "ndcg", "mrr", "map", "precision", "recall", "f1", "accuracy",
    "offline", "online", "a/b test", "experiment", "metric",
    "relevance judgment", "click-through", "conversion"
]

# Explicitly NOT wanted terms
UNWANTED_TERMS = [
    "computer vision", "object detection", "yolo", "gan", "generative adversarial",
    "speech recognition", "tts", "text to speech", "robotics",
    "cad", "solidworks", "fea", "ansys", "mechanical design",
    "brand identity", "packaging design", "creative direction",
    "month-end close", "general ledger", "tax filing", "statutory compliance",
    "warehouse", "fulfillment", "picking", "packing",
    "seo strategy", "content writing", "copywriting"
]

# Cultural signal terms
CULTURAL_TERMS = {
    "decisiveness": ["decided", "migrated", "changed", "replaced", "deprecated",
                     "pivoted", "rebuilt", "overhauled", "chose", "opted"],
    "bias_for_action": ["shipped", "deployed", "launched", "prototype", "mvp",
                        "iteration", "fast", "quickly", "within weeks"],
    "autonomy": ["led", "owned", "designed", "built from scratch", "independently",
                 "sole engineer", "single-handedly", "initiative"],
    "self_awareness": ["i haven't done", "not the core", "self-learner",
                       "i wouldn't call myself", "limited exposure", "lighter on"]
}