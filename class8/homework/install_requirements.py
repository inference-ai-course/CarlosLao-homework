import subprocess
import sys
from pathlib import Path


def install_packages(requirements_filename: str = "requirements.txt") -> None:
    """
    Install Python packages from a requirements file located
    in the same directory as this script.
    """
    # Resolve requirements file relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    req_file = script_dir / requirements_filename

    if not req_file.exists():
        print(f"[ERROR] Requirements file not found: {req_file}")
        sys.exit(1)

    print(f"[INFO] Installing packages from {req_file}...\n")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        )
        print("\n[SUCCESS] All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Package installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_packages()
