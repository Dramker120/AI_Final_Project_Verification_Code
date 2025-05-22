# AI_Final_Project_Verification_Code

## Our Revision Report : [Revision Report](https://docs.google.com/presentation/d/1KrS4LxxDu5PNEDDL7Kq2CilWm1ZMtfKtREevRrvxRUw/edit#slide=id.p)
## Dataset Used Source : [Kaggle - train-num2-var by bhh258](https://www.kaggle.com/datasets/bhh258/train-num2-var)
## 📦 dataset download process
(The dataset used in this repository is not included in the GitHub repository due to size and license constraints. Please follow the instructions below to download it.)

### 1. install Kaggle CLI tools：
   ```bash
   pip install kaggle
   ```
### 2. install kaggle.json (If you've already done this before, just skip this step.)
### 3-1. Manual download
   ```bash
   kaggle datasets download -d bhh258/train-num2-var --unzip
   # You might be prompted to accept the license by typing y.
   ```
### 3-2. Automatic download
   ```bash
      python download_dataset.py
      # The script will handle download, unzip, and license agreement automatically.
   ```