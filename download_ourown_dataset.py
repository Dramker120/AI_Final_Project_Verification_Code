import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import string
from torchvision.utils import save_image
from collections import defaultdict
import math

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

    def get_font_variations(self, font_size):
        """獲取不同的字體變化"""
        fonts = []
        font_paths = [
            "arial.ttf",
            "/System/Library/Fonts/Arial.ttf", 
            "/System/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Arial Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                fonts.append(font)
            except (IOError, OSError):
                continue
        
        if not fonts:
            fonts.append(ImageFont.load_default())
        
        return fonts

    def apply_character_distortion(self, img, char_bbox, distortion_level=0.1):
        """對單個字符應用輕微扭曲"""
        if random.random() < 0.6:  # 60% 機率應用扭曲
            x1, y1, x2, y2 = char_bbox
            # 確保座標是整數
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 檢查邊界框的有效性
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height:
                return img
            
            char_img = img.crop((x1, y1, x2, y2))
            
            # 輕微的透視變換模擬
            width, height = char_img.size
            
            if width <= 0 or height <= 0:
                return img
            
            # 應用輕微的剪切變換
            if random.random() < 0.5:
                # 水平剪切
                shear_factor = random.uniform(-0.2, 0.2)
                transform_matrix = (1, shear_factor, 0, 0, 1, 0)
                try:
                    char_img = char_img.transform(char_img.size, Image.AFFINE, transform_matrix, fillcolor='white')
                    img.paste(char_img, (x1, y1))
                except Exception:
                    # 如果變換失敗，跳過扭曲
                    pass
        
        return img

    def add_noise_effects(self, img, draw):
        """添加各種噪點效果"""
        # 1. 隨機噪點
        noise_density = random.randint(60, 120)
        for _ in range(noise_density):
            x = random.randint(0, self.img_width - 1)
            y = random.randint(0, self.img_height - 1)
            color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
            draw.point((x, y), fill=color)
        
        # 2. 隨機線條
        line_count = random.randint(1, 3)
        for _ in range(line_count):
            start = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            end = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
            width = random.randint(1, 2)
            draw.line([start, end], fill=color, width=width)
        
        # 3. 隨機圓點
        if random.random() < 0.8:  # 80% 機率添加圓點
            dot_count = random.randint(2, 6)
            for _ in range(dot_count):
                x = random.randint(5, self.img_width - 5)
                y = random.randint(5, self.img_height - 5)
                radius = random.randint(1, 3)
                color = (random.randint(160, 200), random.randint(160, 200), random.randint(160, 200))
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

    def add_text_decorations(self, draw, char_bbox, font):
        """添加文字裝飾（刪除線、底線）"""
        x1, y1, x2, y2 = char_bbox
        # 確保座標是整數
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 刪除線 (30% 機率)
        if random.random() < 0.5:
            line_y = y1 + (y2 - y1) // 2
            line_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
            draw.line([(x1, line_y), (x2, line_y)], fill=line_color, width=1)
        
        # 底線 (20% 機率)
        if random.random() < 0.4:
            line_y = y2 - 2
            line_color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
            draw.line([(x1, line_y), (x2, line_y)], fill=line_color, width=1)

    def generate_code_image(self, code):
        img = Image.new('RGB', (self.img_width, self.img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 獲取字體變化
        font_size = random.randint(35, 45)  # 隨機字體大小
        available_fonts = self.get_font_variations(font_size)
        
        char_container_width = self.img_width // self.code_length
        char_bboxes = []  # 儲存每個字符的邊界框
        
        for i, char_text in enumerate(code):
            # 隨機選擇字體
            font = random.choice(available_fonts)
            
            # 計算字符位置
            try:
                char_w = draw.textlength(char_text, font=font)
                char_h = font_size
            except AttributeError:
                char_w, char_h = draw.textsize(char_text, font=font)
            
            base_x = i * char_container_width
            x_offset_in_container = max(0, (char_container_width - char_w) // 2)
            x = base_x + x_offset_in_container + random.randint(-5, 5)
            y_base = (self.img_height - char_h) // 2
            y = y_base + random.randint(-6, 6)
            y = max(0, min(y, self.img_height - char_h))
            
            # 隨機文字顏色（深色系）
            text_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))
            
            # 繪製文字
            draw.text((x, y), char_text, fill=text_color, font=font)
            
            # 記錄字符邊界框
            char_bbox = (int(x), int(y), int(x + char_w), int(y + char_h))
            char_bboxes.append(char_bbox)
            
            # 添加文字裝飾
            self.add_text_decorations(draw, char_bbox, font)
        
        # 應用字符扭曲
        for char_bbox in char_bboxes:
            img = self.apply_character_distortion(img, char_bbox)
        
        # 添加噪點效果
        draw = ImageDraw.Draw(img)  # 重新創建draw對象
        self.add_noise_effects(img, draw)
        
        # 隨機應用輕微模糊 (60% 機率)
        if random.random() < 0.6:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
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