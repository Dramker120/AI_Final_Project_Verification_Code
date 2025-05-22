import os
import subprocess

DATASET = "bhh258/train-num2-var"
DEST_DIR = "data"
CHECK_FILE = os.path.join(DEST_DIR, "train_num2_var", "7","00a7B3J.jpg")

if os.path.exists(CHECK_FILE):
    print(f"Dataset already exists : {CHECK_FILE}")
    print("skipped.")
else:
    print(" start to download dataset...")

    os.makedirs(DEST_DIR, exist_ok=True)

    proc = subprocess.Popen(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", DEST_DIR,
            "--unzip"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    out, err = proc.communicate(input="y\n")  # Automatically enter y to agree to the authorization

    print(out)
    if proc.returncode == 0:
        print("The dataset has been downloaded and unzipped!!!")
    else:
        print(f"Download failed, error message : \n{err}")
