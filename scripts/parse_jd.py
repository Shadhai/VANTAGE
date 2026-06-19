#!/usr/bin/env python3
"""Pre-compute JD requirements. Run once before ranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import JD_FILE, PROCESSED_DIR
from src.parser.jd_parser import JDParser
from src.utils.io_utils import save_json


def make_serializable(obj):
    """Recursively convert dict to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items() 
                if not callable(v) and not k.startswith('_')}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj if not callable(v)]
    elif callable(obj):
        return None  # Skip functions/methods
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def main():
    print("Parsing job description...")
    
    parser = JDParser(JD_FILE)
    requirements = parser.parse()
    
    # Clean non-serializable items
    clean_requirements = make_serializable(requirements)
    
    output_path = PROCESSED_DIR / "jd_requirements.json"
    save_json(clean_requirements, output_path)
    
    print(f"✓ JD requirements saved to {output_path}")
    print(f"  Hard gates: {len(clean_requirements.get('hard_gates', {}))}")
    print(f"  Strong preferences: {len(clean_requirements.get('strong_preferences', {}))}")
    print(f"  Implicit needs: {len(clean_requirements.get('implicit_needs', {}))}")
    print(f"  Weighted terms: {len(clean_requirements.get('weighted_terms', {}))}")


if __name__ == "__main__":
    main()