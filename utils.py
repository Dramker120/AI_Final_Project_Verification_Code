# For Recognizing Verification Code
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import string

CHARS = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase  # 共 63 個字符
char2idx = {ch: idx for idx, ch in enumerate(CHARS)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((60,160)),
            transforms.ToTensor()
        ])
        self.images = images
        self.labels = labels # the indices convert from image name ("load_train_dataset" func handle)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        label=self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)
    
class ValidateDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((60,160)),
            transforms.ToTensor()
        ])
        self.images = images
        self.labels = labels # image name
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
class TestDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((60,160)),
            transforms.ToTensor()
        ])
        self.images = images
        self.labels = labels # image name
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def load_train_dataset(path: str='data/train/')->Tuple[List[str], List[List[int]]]:
  # (TODO) Load training dataset from the given path, return images and labels
    images = []
    labels = []
    for file_name in sorted(os.listdir(path)): # "sorted" is to ensure the order of the images
        if not file_name.endswith('.jpg'):
            continue
        label_str = os.path.splitext(file_name)[0]  # e.g., 'L7AxT'
        if any(c not in char2idx for c in label_str):
            continue  # skip the images contain invalid characters. (Though it shouldn't happen)
        image_path = os.path.join(path, file_name)
        images.append(image_path)
        labels.append([char2idx[c] for c in label_str])
    return images, labels

def load_validate_dataset(path: str='data/val/')->Tuple[List[str], List[str]]:
  # (TODO) Load training dataset from the given path, return images and labels
    images = []
    labels = []
    for file_name in sorted(os.listdir(path)): # "sorted" is to ensure the order of the images
        if not file_name.endswith('.jpg'):
            continue
        label_str = os.path.splitext(file_name)[0]  # e.g., 'L7AxT'
        if any(c not in char2idx for c in label_str):
            continue  # skip the images contain invalid characters. (Though it shouldn't happen)
        image_path = os.path.join(path, file_name)
        images.append(image_path)
        labels.append(label_str)
    return images, labels

def load_test_dataset(path: str='data/test/')->List[str]:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    labels = []
    for file_name in sorted(os.listdir(path)): # "sorted" is to ensure the order of the images
        if not file_name.endswith('.jpg'):
            continue
        label_str = os.path.splitext(file_name)[0]  # e.g., 'L7AxT'
        if any(c not in char2idx for c in label_str):
            continue  # skip the images contain invalid characters. (Though it shouldn't happen)
        image_path = os.path.join(path, file_name)
        images.append(image_path)
        labels.append(label_str)
    return images, labels

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    epochs=range(1,len(train_losses)+1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.show()
    print("Save the plot to 'loss.png'")
    return
def plot2(len_accs: List, val_accs: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    epochs=range(1,len(len_accs)+1)
    plt.plot(epochs, len_accs, label='Length Accuracy')
    plt.plot(epochs, val_accs, label='Validate Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Length and Validate Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('acc.png')
    plt.show()
    print("Save the plot to 'acc.png'")
    return

def plot3(loss_adversarial: List, loss_internal_classes: List, loss_external_classes: List, total_loss: List):
    epochs=range(1,len(loss_adversarial)+1)
    plt.plot(epochs, loss_adversarial, label=' Adversarial Loss')
    plt.plot(epochs, loss_internal_classes, label='Internal Classes Loss')
    plt.plot(epochs, loss_external_classes, label='External Classes Loss')
    plt.plot(epochs, total_loss, label='Total G Loss', linestyle='--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('cGAN : G Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('G_Loss.png')
    plt.show()
    print("Save the plot to 'G_Loss.png'")
    return
