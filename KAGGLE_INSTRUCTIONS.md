# ⚡ Cloud Training Instructions (Kaggle / Colab)

Since local training is slow, follow these steps to train on Kaggle or Google Colab (Free GPU).

### Step 1: Upload the Bundle
1.  Locate the file: `e:\study material\data science project\2nd project\urban_drainage_stress\urban_drainage_kaggle_bundle.zip`
2.  **Kaggle**: Create a new Notebook -> "Add Data" -> "Upload" -> Select the zip file.
3.  **Colab**: Click the "Folder" icon (left sidebar) -> Upload the zip file.

### Step 2: Paste & Run This Code
Copy the code block below into the first cell of your notebook and run it.

```python
import os
import shutil
import sys
import subprocess
import zipfile

# --- STEP 1: INSTALL DEPENDENCIES FIRST (before any directory changes) ---
print("🚀 Installing dependencies (this takes ~2 min)...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torch-geometric", "tqdm", "scipy", "matplotlib", "pandas", "numpy"])
print("✓ Dependencies installed")

# --- CONFIGURATION ---
WORKING_DIR = "/kaggle/working"
PROJECT_DIR = os.path.join(WORKING_DIR, "urban_drainage_stress")

print("\n🔍 Searching for project files...")

# Check possible input directories
INPUT_DIRS = ["/kaggle/input", "../input", "/content"]
input_dir = None
for d in INPUT_DIRS:
    if os.path.exists(d):
        input_dir = d
        break

if input_dir is None:
    print("❌ ERROR: No input directory found! Add data first.")
    sys.exit(1)

# Find project files
print(f"📁 Scanning {input_dir}...")
found_zip = None
found_folder = None

for root, dirs, files in os.walk(input_dir):
    for f in files:
        if f.endswith(".zip"):
            found_zip = os.path.join(root, f)
            break
    if "src" in dirs and "scripts" in dirs:
        found_folder = root
        break
    if found_zip:
        break

# --- STEP 2: COPY/EXTRACT PROJECT ---
if found_zip:
    print(f"📦 Extracting {found_zip}...")
    if os.path.exists(PROJECT_DIR): shutil.rmtree(PROJECT_DIR)
    with zipfile.ZipFile(found_zip, 'r') as zip_ref:
        zip_ref.extractall(PROJECT_DIR)
elif found_folder:
    print(f"📂 Copying from {found_folder}...")
    if os.path.exists(PROJECT_DIR): shutil.rmtree(PROJECT_DIR)
    shutil.copytree(found_folder, PROJECT_DIR)
else:
    print("❌ ERROR: No project files found!")
    sys.exit(1)

print("✓ Project ready")

# --- STEP 3: TRAIN ---
print(f"\n🔥 STARTING TRAINING (500 epochs)...")
os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "scripts/train_latent_model.py"], check=True)

print("\n✅ TRAINING COMPLETE!")
print(f"Download checkpoints from: {PROJECT_DIR}/checkpoints/")
```

### Step 3: Download Results
After training finishes:
1.  Go to `urban_drainage_stress/checkpoints/`
2.  Download `latent_stgnn_best.pt`
3.  Place it back in your local `checkpoints/` folder.
