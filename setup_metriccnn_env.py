import os
import subprocess
import sys

ENV_NAME = "metriccnn"

# Packages compatibles ARM
PACKAGES = [
    "numpy",
    "scikit-image",
    "matplotlib",
    "numba",
    "SimpleITK",
    "torch",
    "torchvision",
    "tqdm",
    "PyYAML",
    "lazy-import",
    "nibabel",
    "itkwidgets",
    "pyvista",
    "nilearn"
]

def run(cmd, fail_ok=False):
    print(cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        if fail_ok:
            return False
        raise
    return True

def conda_exists():
    try:
        subprocess.run(["conda", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def env_exists(name):
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    return any(name in line for line in result.stdout.splitlines())

def main():
    print("=== SETUP ENVIRONMENT METRIC-CNN ===")

    if not conda_exists():
        print("Conda is not installed or not in PATH.")
        sys.exit(1)

    if env_exists(ENV_NAME):
        print(f"Environment '{ENV_NAME}' already exists.")
        print(f"Activate it with: conda activate {ENV_NAME}")
        return

    print(f"Creating environment '{ENV_NAME}'...")
    run(f"conda create -y -n {ENV_NAME} python=3.9")

    print("Installing compatible Python packages...")
    pkg_string = " ".join(PACKAGES)
    run(f"conda run -n {ENV_NAME} pip install {pkg_string}")

    print("Installation complete.")
    print(f"Activate with: conda activate {ENV_NAME}")

if __name__ == "__main__":
    main()
