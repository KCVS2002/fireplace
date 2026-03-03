import subprocess
import sys

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}\n")

packages = ["gymnasium", "numpy", "torch", "stable-baselines3"]
for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print(f"{pkg} done!\n")

print("All packages installed!")
