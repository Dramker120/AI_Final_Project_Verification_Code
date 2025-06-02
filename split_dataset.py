import os
import random
import shutil
import argparse

#Ensure that everyone will get the same train/validate/test dataset
SEED = 42
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2], default=1,
                    help='選擇資料集版本:1=data/train_num2_var/,2=data/dataset2_images/')
args = parser.parse_args()
# 根據 dataset 參數設定 base path
if args.dataset == 1:
    input_base = 'data/train_num2_var' # for Kaggle Dataset
    output_base = 'data' # for Kaggle Dataset
elif args.dataset == 2:
    input_base = 'data/dataset2_images' # for our own Dataset
    output_base = 'data/dataset2_images' # for our own Dataset
for split in ['train', 'val', 'test']:
    #exist_ok : if the folder already exists, do nothing(no error reported).
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

for length in range(4, 8):  # 4~7
    folder_path = os.path.join(input_base, str(length))
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')] # jpg for Kaggle Dataset, png for our own Dataset
    # check
    check_path = os.path.join(output_base, 'train', f'{filenames[0]}')
    if os.path.exists(check_path):
        print(f'Skipping length={length}, seems already processed.')
        continue

    random.shuffle(filenames)
    n = len(filenames)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    split_files = {
        'train': filenames[:n_train],
        'val': filenames[n_train:n_train + n_val],
        'test': filenames[n_train + n_val:]
    }

    for split, files in split_files.items():
        for fname in files:
            src = os.path.join(folder_path, fname)
            dst = os.path.join(output_base, split, f'{fname}')  # 防重名
            shutil.copy(src, dst)

