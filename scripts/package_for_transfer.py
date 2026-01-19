#!/usr/bin/env python3
"""
Package Snowcrash for efficient transfer to Old Nano via SCP.

Packaging strategy:
1. Compress models (GGUF already compressed, but can tar)
2. Exclude unnecessary files (.git, __pycache__, .pyc, etc.)
3. Create deployment package
4. Calculate size and verify fits Old Nano constraints
"""
import os
import subprocess
import tarfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "package"
PACKAGE_NAME = f"snowcrash-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"

# Files/directories to exclude from package
EXCLUDE_PATTERNS = [
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "*.log",
    ".DS_Store",
    "*.swp",
    "*.swo",
    "*~",
    "tests/__pycache__",
    "models/*/__pycache__"
]

def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # Skip excluded patterns
        skip = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in dirpath or any(pattern.replace("*", "") in f for f in filenames):
                skip = True
                break
        if skip:
            continue
            
        for filename in filenames:
            filepath = Path(dirpath) / filename
            if filepath.exists():
                total += filepath.stat().st_size
    return total

def create_package():
    """Create deployment package."""
    print("Creating deployment package for Old Nano...")
    print(f"Source: {PROJECT_ROOT}")
    print(f"Package: {PACKAGE_NAME}\n")
    
    # Create package directory
    PACKAGE_DIR.mkdir(exist_ok=True)
    package_path = PACKAGE_DIR / PACKAGE_NAME
    
    # Calculate size before packaging
    total_size = 0
    
    # Include directories
    include_dirs = [
        "models",
        "mcp_server",
        "tools",
        "scripts"
    ]
    
    include_files = [
        "requirements.txt",
        "README.md"
    ]
    
    print("Including:")
    for dirname in include_dirs:
        dirpath = PROJECT_ROOT / dirname
        if dirpath.exists():
            size = get_dir_size(dirpath)
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  {dirname}/ ({size_mb:.1f} MB)")
    
    for filename in include_files:
        filepath = PROJECT_ROOT / filename
        if filepath.exists():
            size = filepath.stat().st_size
            total_size += size
            print(f"  {filename}")
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\nTotal size (before compression): {total_size_mb:.1f} MB")
    
    # Create tar.gz
    print(f"\nCreating compressed package: {package_path}")
    with tarfile.open(package_path, "w:gz") as tar:
        # Add directories
        for dirname in include_dirs:
            dirpath = PROJECT_ROOT / dirname
            if dirpath.exists():
                tar.add(dirpath, arcname=dirname, filter=lambda tarinfo: exclude_filter(tarinfo))
        
        # Add files
        for filename in include_files:
            filepath = PROJECT_ROOT / filename
            if filepath.exists():
                tar.add(filepath, arcname=filename)
    
    # Get compressed size
    compressed_size = package_path.stat().st_size
    compressed_size_mb = compressed_size / (1024 * 1024)
    
    print(f"Compressed size: {compressed_size_mb:.1f} MB")
    print(f"Compression ratio: {compressed_size / total_size * 100:.1f}%")
    
    # SCP command suggestion
    print("\n" + "="*50)
    print("Package created successfully!")
    print(f"\nTo transfer to Old Nano:")
    print(f"  scp {package_path} old-nano:~/snowcrash/")
    print(f"\nOn Old Nano, extract with:")
    print(f"  tar -xzf {PACKAGE_NAME}")
    
    return package_path

def exclude_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Filter out excluded files."""
    name = tarinfo.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern in name:
            return None
    return tarinfo

def main():
    """Main packaging function."""
    package_path = create_package()
    print(f"\n[OK] Package ready: {package_path}")

if __name__ == "__main__":
    main()

