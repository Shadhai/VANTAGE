@echo off
echo ============================================
echo VANTAGE - Candidate Ranking System
echo ============================================
echo.

REM Quick check for torch
python -c "import torch" 2>nul
if %errorlevel% neq 0 (
    echo ⚠ PyTorch not working. Running Windows setup...
    python scripts/setup_windows.py
    if %errorlevel% neq 0 (
        echo Setup failed. Please fix PyTorch manually.
        echo https://aka.ms/vs/17/release/vc_redist.x64.exe
        pause
        exit /b 1
    )
)
echo.

REM Rest of pipeline...
python scripts/parse_jd.py
python scripts/precompute_embeddings.py
python scripts/precompute_features.py
python scripts/train_ranker.py
python src/pipeline.py --candidates data/raw/candidates.jsonl --jd data/raw/job_description.md --output output/submission.csv

echo.
echo Done! Output: output\submission.csv
pause