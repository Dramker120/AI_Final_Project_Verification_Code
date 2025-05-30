# train_cgan.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import string
import random

# -------------------------
# Config
# -------------------------
IMG_HEIGHT = 60
IMG_WIDTH = 160
MAX_LEN = 7
EMBED_DIM = 32
BATCH_SIZE = 64
EPOCHS = 10
NOISE_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARS = string.ascii_uppercase + string.digits
char2id = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# -------------------------
# Utilities
# -------------------------
def text_to_ids(text):
    ids = [char2id.get(c, 0) for c in text]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids[:MAX_LEN])

# -------------------------
# Dataset
# -------------------------
class CaptchaDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith(".jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT , IMG_WIDTH )),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = os.path.splitext(fname)[0]
        text_ids = text_to_ids(label)
        image = Image.open(os.path.join(self.root, fname)).convert("L")
        return self.transform(image), text_ids

# -------------------------
# Models
# -------------------------
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM * MAX_LEN, 128)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, IMG_HEIGHT * IMG_WIDTH),
            nn.Tanh()
        )

    def forward(self, noise, text_emb):
        x = torch.cat([noise, text_emb], dim=1)
        x = self.fc(x)
        return x.view(-1, 1, IMG_HEIGHT, IMG_WIDTH)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(IMG_HEIGHT * IMG_WIDTH + 128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_emb):
        x = img.view(img.size(0), -1)
        x = torch.cat([x, text_emb], dim=1)
        return self.fc(x)

# -------------------------
# Training Loop
# -------------------------
def train():
    dataset = CaptchaDataset("./data/train")  #
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    TE = TextEncoder().to(DEVICE)

    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(list(G.parameters()) + list(TE.parameters()), lr=0.0002)
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        for i, (real_imgs, text_ids) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            text_ids = text_ids.to(DEVICE)
            text_emb = TE(text_ids)

            valid = torch.ones(batch_size, 1, device=DEVICE)
            fake = torch.zeros(batch_size, 1, device=DEVICE)

            # Train Generator
            optim_G.zero_grad()
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            gen_imgs = G(noise, text_emb)
            pred = D(gen_imgs, text_emb)
            g_loss = criterion(pred, valid)
            g_loss.backward()
            optim_G.step()

            # Train Discriminator
            optim_D.zero_grad()
            real_pred = D(real_imgs, text_emb.detach())
            d_real_loss = criterion(real_pred, valid)
            fake_pred = D(gen_imgs.detach(), text_emb.detach())
            d_fake_loss = criterion(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        torch.save(G.state_dict(), f"generator_epoch{epoch}.pt")

if __name__ == "__main__":
    train()
