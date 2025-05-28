import os
import random
import shutil

#Ensure that everyone will get the same train/validate/test dataset
SEED = 42
random.seed(SEED)

input_base = 'data/train_num2_var'
output_base = 'data'

for split in ['train', 'val', 'test']:
    #exist_ok : if the folder already exists, do nothing(no error reported).
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

for length in range(4, 8):  # 4~7
    folder_path = os.path.join(input_base, str(length))
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
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

