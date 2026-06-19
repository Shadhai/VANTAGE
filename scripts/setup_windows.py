#!/usr/bin/env python3
"""
Windows Setup for VANTAGE
Fixes PyTorch DLL issues and installs all dependencies correctly.
"""

import sys
import subprocess
import platform
from pathlib import Path

def check_admin():
    """Check if running with admin privileges."""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_vc_redist():
    """Check if Visual C++ Redistributable is installed."""
    import winreg
    
    vc_keys = [
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    ]
    
    for key_path in vc_keys:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            winreg.CloseKey(key)
            return True
        except WindowsError:
            continue
    
    return False

def main():
    print("=" * 60)
    print("VANTAGE - Windows Setup")
    print("=" * 60)
    
    if platform.system() != "Windows":
        print("This script is for Windows only.")
        print("For Linux/Mac, use: bash run.sh")
        return
    
    # Check VC++ Redistributable
    print("\n[1/4] Checking Visual C++ Redistributable...")
    if check_vc_redist():
        print("  ✓ VC++ Redistributable found")
    else:
        print("  ✗ VC++ Redistributable NOT FOUND")
        print("\n  This is required for PyTorch to work on Windows.")
        print("  Download and install from:")
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n  After installing, re-run this script.")
        
        response = input("\n  Continue without VC++? PyTorch may fail. (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install PyTorch CPU-only (smaller download, no CUDA needed)
    print("\n[2/4] Installing PyTorch (CPU-only for smaller download)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("  ✓ PyTorch installed")
    except subprocess.CalledProcessError:
        print("  ✗ PyTorch installation failed")
        print("  Try: pip install torch")
        sys.exit(1)
    
    # Test PyTorch
    print("\n[3/4] Testing PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__} works!")
        
        # Quick tensor test
        t = torch.randn(2, 3)
        print(f"  ✓ Tensor operations work: {t.shape}")
    except Exception as e:
        print(f"  ✗ PyTorch error: {e}")
        print("\n  This usually means VC++ Redistributable is missing.")
        print("  Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        sys.exit(1)
    
    # Install sentence-transformers
    print("\n[4/4] Installing sentence-transformers...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "sentence-transformers"
        ])
        print("  ✓ sentence-transformers installed")
    except subprocess.CalledProcessError:
        print("  ✗ Installation failed")
        sys.exit(1)
    
    # Final test
    print("\n" + "=" * 60)
    print("Testing full ML stack...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  ✓ sentence-transformers imported")
        
        # Quick encode test
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["test sentence"])
        print(f"  ✓ Embedding generated: shape {test_embedding.shape}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Windows setup complete!")
    print("  PyTorch + sentence-transformers working correctly")
    print("\n  Run the pipeline:")
    print("    python scripts/precompute_embeddings.py")
    print("    python run.bat")
    print("=" * 60)

if __name__ == "__main__":
    main()