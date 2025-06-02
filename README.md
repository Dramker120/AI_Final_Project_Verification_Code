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
### 3-1. Download Kaggle Dataset
#### 3-1-1. Manual download
   ```bash
   kaggle datasets download -d bhh258/train-num2-var --unzip
   # You might be prompted to accept the license by typing "y" or "yes".
   ```
#### 3-1-2. Automatic download (recommend)
   ```bash
      python download_kaggle_dataset.py
      # The script will handle download, unzip, and license agreement automatically.
   ```
### 3-2.  Download our own Dataset
#### Automatic download (recommend)
   ```bash
      python download_ourown_dataset.py
      # The script will handle download, unzip, and license agreement automatically.
   ```
### 4-1. Split dataset (Kaggle Dataset)
   ```bash
      python split_dataset.py --dataset 1
      # This script will automatically split the downloaded dataset into training, validation, and test sets (by length and with a fixed seed for reproducibility).
   ```
### 4-2. Split dataset (Our own Dataset)
   ```bash
      python split_dataset.py --dataset 2
      # This script will automatically split the downloaded dataset into training, validation, and test sets (by length and with a fixed seed for reproducibility).
   ```
##  Training a model for recognizing verification codes
### 1. install environment
We recommend using Python 3.8 or above.
#### Use GPU (recommend)
Please install the correct version of PyTorch that matches your system’s CUDA version (instructions not provided here).
#### Using CPU Only (not recommend)
You can use :
```bash
pip install torch torchvision torchaudio
```
#### Other libraries
All other required libraries are listed in "requirements.txt".
You can install them with the following command :
```bash
pip install -r requirements.txt
```
### 2-1. Start Recognizing (Use Kaggle Dataset)
```bash
python Recognize.py --dataset 1
# Average runtime: approximately 1 hour
```
### Results: Test accuracy is about 84%
![Test result](Recognition_result.png)

### 2-2. Start Recognizing (Use our own Dataset)
```bash
python Recognize.py --dataset 2
# Average runtime: approximately 30 minutes
```
### Results: Test accuracy is about 99%
![Test result](Recognition_result2.png)

##  Generating CAPTCHA images 
### Note
The captcha in the filename now matches the actual image captcha, but it can be further improved.
### 1. Train cGAN model
```bash
python cGAN.py --n_epochs 100 --save_model # You can resize n_epochs by changing '100' to other numbers.
```
### 2. Generate datasets for recognizing
```bash
python cGAN.py --generate_only --load_model models/cgan_final.pth
```


