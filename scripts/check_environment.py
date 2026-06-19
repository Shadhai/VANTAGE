#!/usr/bin/env python3
"""Check that the environment is properly set up before running."""

import sys
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CANDIDATES_FILE, JD_FILE, OUTPUT_DIR


def check_package(import_name: str, package_name: str = None) -> bool:
    """Safely check if a package is importable."""
    if package_name is None:
        package_name = import_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False
    except Exception as e:
        # Package exists but has runtime issues (e.g., DLL errors)
        print(f"  ⚠ {package_name} - INSTALLED but has issues: {type(e).__name__}")
        return False


def main():
    print("=" * 60)
    print("VANTAGE Environment Check")
    print("=" * 60)
    
    all_ok = True
    warnings = []
    
    # Python version
    print(f"\nPython: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"RAM: {ram_gb:.1f} GB")
        if ram_gb < 8:
            warnings.append("Low RAM. 16GB recommended but 8GB may work.")
    except ImportError:
        print("RAM: Unknown (pip install psutil for memory check)")
    
    # Check CPU cores
    cpu_count = None
    try:
        import os
        cpu_count = os.cpu_count()
        print(f"CPU cores: {cpu_count}")
    except:
        print("CPU cores: Unknown")
    
    # Check required packages
    print("\nChecking required packages...")
    
    # Core (no heavy dependencies)
    core_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'orjson': 'orjson',
        'pyarrow': 'pyarrow',
        'tqdm': 'tqdm',
    }
    
    for import_name, package_name in core_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED (pip install {package_name})")
            all_ok = False
    
    # Check sentence-transformers and torch separately (heavy, may fail on Windows)
    print("\nChecking ML packages...")
    
    torch_ok = False
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        torch_ok = True
    except ImportError:
        print(f"  ✗ torch - NOT INSTALLED (pip install torch)")
        all_ok = False
    except OSError as e:
        print(f"  ⚠ torch - DLL ERROR on Windows")
        print(f"    Error: {e}")
        print(f"    Fix: Install Microsoft Visual C++ Redistributable")
        print(f"    https://aka.ms/vs/17/release/vc_redist.x64.exe")
        warnings.append("PyTorch DLL error - install VC++ Redistributable")
    
    # Try sentence-transformers only if torch is OK
    if torch_ok:
        try:
            import sentence_transformers
            print(f"  ✓ sentence-transformers")
        except ImportError:
            print(f"  ✗ sentence-transformers - NOT INSTALLED")
            all_ok = False
    else:
        # Try importing anyway to see specific error
        try:
            import sentence_transformers
            print(f"  ✓ sentence-transformers (torch issue resolved at import)")
        except ImportError:
            print(f"  ⚠ sentence-transformers - CANNOT IMPORT (torch dependency)")
            warnings.append("sentence-transformers unavailable - will use TF-IDF fallback")
        except OSError:
            print(f"  ⚠ sentence-transformers - DLL ERROR")
            warnings.append("sentence-transformers unavailable - will use TF-IDF fallback")
    
    # Check data files
    print("\nChecking data files...")
    if CANDIDATES_FILE.exists():
        size_mb = CANDIDATES_FILE.stat().st_size / (1024**2)
        print(f"  ✓ candidates.jsonl ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ candidates.jsonl NOT FOUND")
        print(f"    Expected at: {CANDIDATES_FILE}")
        all_ok = False
    
    if JD_FILE.exists():
        print(f"  ✓ job_description.md")
    else:
        print(f"  ✗ job_description.md NOT FOUND")
        print(f"    Expected at: {JD_FILE}")
        all_ok = False
    
    # Check directories
    print("\nChecking directories...")
    dirs_to_check = [
        OUTPUT_DIR,
        Path("logs"),
        Path("data/processed"),
        Path("models"),
    ]
    for dir_path in dirs_to_check:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        free_gb = free / (1024**3)
        print(f"\nDisk space free: {free_gb:.1f} GB")
        if free_gb < 2:
            warnings.append(f"Low disk space ({free_gb:.1f} GB). Need ~2GB for pre-computed files.")
    except:
        pass
    
    # Check for validate_submission.py
    validator_path = Path("scripts/validate_submission.py")
    if validator_path.exists():
        print(f"  ✓ validate_submission.py found")
    else:
        print(f"  ⚠ validate_submission.py not found in scripts/")
        print(f"    Copy it from the hackathon bundle to scripts/")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ Environment is ready!")
        if warnings:
            print(f"\n⚠ {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  - {w}")
        print("\n  Run: bash run.sh")
        print("   Or: python src/pipeline.py")
        return 0
    else:
        print("✗ Some checks failed. Fix the issues above.")
        if warnings:
            print(f"\n⚠ {len(warnings)} additional warning(s):")
            for w in warnings:
                print(f"  - {w}")
        print("\n  Fix errors: pip install -r requirements.txt")
        print("  Fix torch DLL: install VC++ Redistributable")
        return 1


if __name__ == "__main__":
    sys.exit(main())