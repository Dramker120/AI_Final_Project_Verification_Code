# For Recognizing Verification Code
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from sklearn.utils import shuffle
from loguru import logger
# from sklearn.metrics import accuracy_score

from CNNCTC import CNNCTC, train, validate, test
from utils import TrainDataset, ValidateDataset, TestDataset, load_train_dataset, load_validate_dataset, load_test_dataset, plot, plot2

"""
Notice:
    1) You can't add any additional package
    2) You can ignore the suggested data type if you want
"""

def ctc_collate_fn(batch): # only train need since 
    images, labels = zip(*batch)
    return torch.stack(images), labels  # 不對 label 做 padding

def main():
    """
    load data
    """
    logger.info("Start loading training data")
    train_images, train_labels = load_train_dataset()
    #train_images = train_images[:8000] # fast test
    #train_labels = train_labels[:8000] # fast test
    logger.info("Start loading validate data")
    val_images, val_labels = load_validate_dataset()
    #val_images = val_images[:1000] # fast test
    #val_labels = val_labels[:1000] # fast test
    logger.info("Start loading test data")
    test_images, test_labels = load_test_dataset()
    
    train_dataset = TrainDataset(train_images, train_labels)
    val_dataset = ValidateDataset(val_images, val_labels)
    test_dataset = TestDataset(test_images, test_labels)

    #CNN - train and validate
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
    val_chars_accuracy = []
    max_acc = 0

    EPOCHS = 30
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_len_acc, val_char_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_length_accuracy.append(val_len_acc)
        val_chars_accuracy.append(val_char_acc)
        max_acc=max(max_acc,val_acc)
        # (TODO) Print the training log to help you monitor the training process
        #        You can save the model for future usage
        print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"Train Loss = {train_loss:.4f}, "
          f"Val Loss = {val_loss:.4f}, "
          f"Val Len Acc = {val_len_acc:.2%}, "
          f"Val Char Acc = {val_char_acc:.2%}, "
          f"Val Acc = {val_acc:.2%}")

    logger.info(f"Best Accuracy: {max_acc:.4f}")

    
    #CNN - plot
    
    plot(train_losses, val_losses)
    plot2(val_length_accuracy, val_chars_accuracy)
    
    #CNN - test
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=6,pin_memory=True,persistent_workers=True)
    test_accuracy = test(model, test_loader, criterion, device)
    print(f"Test accuracy = {test_accuracy:.3%}")

if __name__ == '__main__':
    main()
