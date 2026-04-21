import subprocess, sys

packages = ["datasets", "huggingface_hub"]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    print(f"✓ {pkg} kuruldu")