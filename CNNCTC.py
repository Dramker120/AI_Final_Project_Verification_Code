# For Recognizing Verification Code
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn # PyTorch的神經網路模組庫
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd
import csv
import os
from utils import char2idx, idx2char# validate function need them to compute loss and accuracy

class DynamicDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def set_p(self, new_p):
        self.p = new_p

    def forward(self,x):
        return F.dropout(x, p=self.p, training=self.training)

class CNNCTC(nn.Module):
    def __init__(self, num_classes=63): # 1+10+26*2 (1 for "blank")
        super(CNNCTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # input: (B, 3, 60, 160)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # No maxpool to try get subtle features.

            nn.Conv2d(64, 256, kernel_size=3, padding=1), # input: (B, 64, 60, 160)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # (B, 256, 30, 80)

            nn.Conv2d(256, 128, kernel_size=1), # input: (B, 256, 30, 80)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # input: (B, 128, 30, 80)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=1), # input: (B, 128, 30, 80)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # input: (B, 256, 30, 80)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (B, 256, 15, 40)
        )
        self.dropout=DynamicDropout(p=0.3)
        self.linear = nn.Linear(256 * 15, num_classes)  # 把高度整合進來，變成序列每一步的輸出

    def forward(self, x):  # x: (B, 1, 60, 160)
        x = self.cnn(x)     # (B, 256, 15, 40)
        x = x.permute(3, 0, 1, 2)  # -> (T=40, B, C=256, H=15)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # (T, B, 256*15)
        x=self.dropout(x)   # second method in experiment
        x = self.linear(x)  # (T, B, num_classes)
        return x
    
def train(model: CNNCTC, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)

        # 準備 targets 和 target_lengths
        target_lengths = [len(label) for label in labels]
        targets = [c for label in labels for c in label]  # flatten
        targets = torch.tensor(targets, dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

        # forward
        outputs = model(inputs)  # shape = (T, B, num_classes)
        log_probs = outputs.log_softmax(2)  # for CTC loss , dim=2 表示沿著 num_classes 這一維做 softmax
        T, B, _ = log_probs.size() # get input length

        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)  # 每個序列都長度為 T

        # loss 計算
        loss = criterion(log_probs, targets, input_lengths, target_lengths) # criterion = nn.CTCLoss(blank=0)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def ctc_greedy_decoder(log_probs: torch.Tensor, blank: int = 0) -> list[list[int]]:
    # log_probs: (T, B, num_classes)
    max_probs = log_probs.argmax(dim=2)  # (T, B)
    max_probs = max_probs.transpose(0, 1)  # (B, T) Why do this? Since later we will deal data by batches not by time 

    results = []
    for seq in max_probs:
        decoded = []
        prev = blank
        for token in seq:
            token = token.item()
            if token != blank and token != prev:
                decoded.append(token)
            prev = token
        results.append(decoded)
    return results

def validate(model: CNNCTC, val_loader: DataLoader, criterion, device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    length_accuracy = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)

            # 準備 targets
            target_lengths = [len(label) for label in labels]
            targets = [char2idx[c] for label in labels for c in label]
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            # forward
            outputs = model(inputs)  # shape = (T, B, num_classes)
            log_probs = outputs.log_softmax(2)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), T, dtype=torch.long)

            # compute loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # decode and compute accuracy
            pred_seqs = ctc_greedy_decoder(log_probs)  # list of list[int]
            pred_strs = ["".join([idx2char[i] for i in seq]) for seq in pred_seqs]
            label_strs = [label for label in labels]

            for pred, label in zip(pred_strs, label_strs):
                if pred == label:
                    total_correct += 1
                if len(pred) == len(label):
                    length_accuracy += 1
            total_samples += len(labels)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    length_accuracy /= total_samples
    return avg_loss, accuracy, length_accuracy

def test(model: CNNCTC, val_loader: DataLoader, criterion, device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)

            # forward
            outputs = model(inputs)  # shape = (T, B, num_classes)
            log_probs = outputs.log_softmax(2)

            # decode and compute accuracy
            pred_seqs = ctc_greedy_decoder(log_probs)  # list of list[int]
            pred_strs = ["".join([idx2char[i] for i in seq]) for seq in pred_seqs]
            label_strs = [label for label in labels]

            for pred, label in zip(pred_strs, label_strs):
                if pred == label:
                    total_correct += 1
            total_samples += len(labels)

    accuracy = total_correct / total_samples
    return accuracy