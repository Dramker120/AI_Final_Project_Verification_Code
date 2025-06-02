# For Recognizing Verification Code
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from sklearn.utils import shuffle
from loguru import logger
# from sklearn.metrics import accuracy_score
import argparse

from CNNCTC import CNNCTC, train, validate, test
from utils import TrainDataset, ValidateDataset, TestDataset, load_train_dataset, load_validate_dataset, load_test_dataset, plot, plot2

"""
Notice:
    1) You can't add any additional package
    2) You can ignore the suggested data type if you want
"""

def ctc_collate_fn(batch): # only train need since SS
    images, labels = zip(*batch)
    return torch.stack(images), labels  # 不對 label 做 padding

def dynamic_p(epoch, stationary = True):
    if stationary: return 0.3
    return 0.1+1/( (epoch//7) + 2)

def main():
    """
    load data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                        help='選擇資料集版本:1=data/,2=data/dataset2_images/,3=data/dataset3_images/')
    args = parser.parse_args()

    # 根據 dataset 參數設定 base path
    if args.dataset == 1:
        base_path_train = 'data/train'
        base_path_val = 'data/val'
        base_path_test = 'data/test'
    elif args.dataset == 2:
        base_path_train = 'data/dataset2_images/train'
        base_path_val = 'data/dataset2_images/val'
        base_path_test = 'data/dataset2_images/test'
    elif args.dataset == 3:
        base_path_train = 'data/dataset3_images/train'
        base_path_val = 'data/dataset3_images/val'
        base_path_test = 'data/dataset3_images/test'
    logger.info(f"Training use path: {base_path_train}")
    logger.info("Start loading training data")
    train_images, train_labels = load_train_dataset(base_path_train)
    #train_images = train_images[:8000] # fast test
    #train_labels = train_labels[:8000] # fast test
    logger.info(f"Training use path: {base_path_val}")
    logger.info("Start loading validate data")
    val_images, val_labels = load_validate_dataset(base_path_val)
    #val_images = val_images[:1000] # fast test
    #val_labels = val_labels[:1000] # fast test
    logger.info(f"Training use path: {base_path_test}")
    logger.info("Start loading test data")
    test_images, test_labels = load_test_dataset(base_path_test)
    
    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = ValidateDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images, test_labels)

    #CNNCTC - train and validate
    logger.info("Start training CNNCTC")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=6,pin_memory=True,persistent_workers=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=6,pin_memory=True,persistent_workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNCTC().to(device)
    criterion = nn.CTCLoss(blank=0)  # In utils, we make char2idx[0] = '-' => blank=0

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)

    train_losses = []
    val_losses = []
    val_length_accuracy = []
    val_accuracy = []
    max_acc = 0


    EPOCHS = 100
    record_times = 0
    for epoch in range(EPOCHS): #epoch
        new_p = dynamic_p(epoch,stationary = False)
        model.dropout.set_p(new_p)
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_len_acc = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_length_accuracy.append(val_len_acc)
        val_accuracy.append(val_acc)
        if val_acc < max_acc+0.01: record_times += 1
        else: record_times = 0
        max_acc=max(max_acc,val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Dropout p = {new_p:.4f}, "
          f"Train Loss = {train_loss:.4f}, "
          f"Val Loss = {val_loss:.4f}, "
          f"Val Len Acc = {val_len_acc:.2%}, "
          f"Val Acc = {val_acc:.2%}")
        if record_times == 10: 
            print(f"At Epoch {epoch+1}, guess that overfitting may happen!")
            break

    logger.info(f"Best Accuracy: {max_acc:.4f}")
    
    #CNNCTC - plot
    
    plot(train_losses, val_losses)
    plot2(val_length_accuracy, val_accuracy)
    
    #CNNCTC - test
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=6,pin_memory=True,persistent_workers=True)
    test_accuracy = test(model, test_loader, criterion, device)
    print(f"Test accuracy = {test_accuracy:.3%}")

if __name__ == '__main__':
    main()
