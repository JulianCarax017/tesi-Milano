"""Print installed package versions without importing ML packages or running models.

Run manually in the existing experiment environment to capture its versions.
"""
import importlib.metadata
import platform
import sys

PACKAGES = (
    "torch", "transformers", "tokenizers", "huggingface-hub", "datasets",
    "numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib", "tqdm",
    "jupyterlab", "ipykernel", "ipython", "nbconvert", "nbformat",
)

def main():
    print(f"# Python {platform.python_version()}")
    print(f"# Platform: {platform.system()} {platform.machine()}")
    missing = []
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
        else:
            print(f"{package}=={version}")
    if missing:
        print("# Not installed: " + ", ".join(missing), file=sys.stderr)

if __name__ == "__main__":
    main()
