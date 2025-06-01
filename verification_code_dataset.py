# verification_code_dataset.py

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import string
from torchvision.utils import save_image
from collections import defaultdict

# 你需要提前定義 CHARS, char2idx
CHARS = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase  # 共 63 個字符
char2idx = {ch: idx for idx, ch in enumerate(CHARS)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

class VerificationCodeDataset(Dataset):
    def __init__(self, size=10000, code_length=5, img_height=60, img_width=160, seed=42):
        self.size = size
        self.code_length = code_length
        self.img_height = img_height
        self.img_width = img_width
        self.seed = seed

        # 為了可重現：先固定亂數種子，產生所有code
        random.seed(self.seed)
        self.codes = [''.join(random.choices(CHARS[1:], k=code_length)) for _ in range(size)]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def generate_code_image(self, code):
        img = Image.new('RGB', (self.img_width, self.img_height), 'white')
        draw = ImageDraw.Draw(img)
        font_size = 40
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

        char_container_width = self.img_width // self.code_length
        for i, char_text in enumerate(code):
            try:
                char_w = draw.textlength(char_text, font=font)
            except AttributeError:
                char_w, _ = draw.textsize(char_text, font=font)
            base_x = i * char_container_width
            x_offset_in_container = max(0, (char_container_width - char_w) // 2)
            x = base_x + x_offset_in_container + random.randint(-3, 3)
            y_base = (self.img_height - font_size) // 2
            y = y_base + random.randint(-4, 4)
            y = max(0, min(y, self.img_height - font_size))
            draw.text((x, y), char_text, fill=(0, 0, 0), font=font)

        for _ in range(random.randint(0, 1)):
            start = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            end = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            draw.line([start, end], fill=(random.randint(190, 225), random.randint(190, 225), random.randint(190, 225)), width=random.randint(1, 2))
        for _ in range(random.randint(40, 80)):
            draw.point((random.randint(0, self.img_width - 1), random.randint(0, self.img_height - 1)),
                       fill=(random.randint(170, 210), random.randint(170, 210), random.randint(170, 210)))
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        code = self.codes[idx]
        img = self.generate_code_image(code)
        if self.transform:
            img = self.transform(img)
        code_indices = [char2idx[c] for c in code]
        return img, torch.tensor(code_indices, dtype=torch.long)
    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        label_counts = defaultdict(int)

        for idx in range(self.size):
            code = self.codes[idx]
            img = self.generate_code_image(code)
            img_tensor = self.transform(img) if self.transform else img

            label_counts[code] += 1
            suffix = f"_{label_counts[code]}" if label_counts[code] > 1 else ""
            filename = f"{code}{suffix}.png"
            path = os.path.join(folder, filename)

            save_image(img_tensor, path)
        print(f"Saved {self.size} images to {folder}")

if __name__ == "__main__":
    for length in range(4,8): 
        dataset = VerificationCodeDataset(size=10000,code_length=length, seed=42)
        dataset.save_images(f"data/dataset2_images/{length}/")
