import os
import subprocess

DATASET = "bhh258/train-num2-var"
DEST_DIR = "data"
CHECK_FILE = os.path.join(DEST_DIR, "train_num2_var", "7","00a7B3J.jpg")

if os.path.exists(CHECK_FILE):
    print(f"✅ 已偵測到資料集已存在：{CHECK_FILE}")
    print("⏩ 跳過下載。")
else:
    print("📥 開始下載資料集...")

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

    out, err = proc.communicate(input="y\n")  # 自動輸入 y 同意授權

    print(out)
    if proc.returncode == 0:
        print("✅ 資料集下載並解壓縮完成！")
    else:
        print(f"❌ 下載失敗，錯誤訊息:\n{err}")
