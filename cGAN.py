import argparse
import os
import numpy as np
import math
import string
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# 定義字符集（與utils.py保持一致）
CHARS = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase  # 共 63 個字符
char2idx = {ch: idx for idx, ch in enumerate(CHARS)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

os.makedirs("images", exist_ok=True)
os.makedirs("generated_data/train", exist_ok=True)
os.makedirs("models", exist_ok=True)  # 新增：保存模型的目錄

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--code_length", type=int, default=5, help="length of verification code")
parser.add_argument("--img_height", type=int, default=60, help="height of each image")
parser.add_argument("--img_width", type=int, default=160, help="width of each image")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--generate_dataset", action='store_true', help="generate dataset for training")
parser.add_argument("--dataset_size", type=int, default=10000, help="size of generated dataset")
# 新增參數
parser.add_argument("--generate_only", action='store_true', help="只生成數據集，不進行訓練")
parser.add_argument("--load_model", type=str, default="", help="載入已訓練的模型路徑")
parser.add_argument("--save_model", action='store_true', help="保存訓練完成的模型")
parser.add_argument("--generate_count", type=int, default=20000, help="要生成的驗證碼圖片數量")
parser.add_argument("--output_dir", type=str, default="generated_data/train", help="生成圖片的輸出目錄")
parser.add_argument("--lambda_cls", type=float, default=10.0, help="Weight for classification loss in generator") # 新增：分類損失的權重

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_height, opt.img_width)
cuda = True if torch.cuda.is_available() else False

class VerificationCodeDataset(Dataset):
    """生成驗證碼樣本的dataset"""
    def __init__(self, size=10000, code_length=5, img_height=60, img_width=160):
        self.size = size
        self.code_length = code_length
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 歸一化到 [-1, 1]
        ])

    def generate_code_image(self, code):
        """生成更清晰的驗證碼圖片"""
        # 創建白色背景
        img = Image.new('RGB', (self.img_width, self.img_height), 'white')
        draw = ImageDraw.Draw(img)

        # 嘗試使用系統字體，並增大字體
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
            # 獲取字符實際寬度
            try:
                # Pillow 9.2.0+
                char_w = draw.textlength(char_text, font=font)
            except AttributeError:
                # 舊版 Pillow
                char_w, _ = draw.textsize(char_text, font=font)

            # 在字符容器內居中並添加少量抖動
            base_x = i * char_container_width
            # 確保 x_offset_in_container 不為負，如果 char_w 大於 container_width
            x_offset_in_container = max(0, (char_container_width - char_w) // 2)

            x = base_x + x_offset_in_container + random.randint(-3, 3) # 減小 x 軸隨機抖動
            # 調整y軸位置，確保字符在圖片內且有一定隨機性
            # 假設字符大致高度為 font_size
            y_base = (self.img_height - font_size) // 2 # 基礎居中位置
            y = y_base + random.randint(-4, 4) # 輕微垂直抖動
            y = max(0, min(y, self.img_height - font_size)) # 確保不出界

            # 使用純黑色字符以獲得最大對比度
            char_color = (0, 0, 0)

            # 繪製字符
            draw.text((x, y), char_text, fill=char_color, font=font)

        # 添加少量且顏色較淺的噪聲線條
        for _ in range(random.randint(0, 1)): # 0到1條線
            start = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            end = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            draw.line([start, end],
                      fill=(random.randint(190, 225), random.randint(190, 225), random.randint(190, 225)), # 更淺的線條
                      width=random.randint(1,2))

        # 添加顏色較淺的噪點
        for _ in range(random.randint(40, 80)): # 噪點數量可以根據效果調整
            x_p = random.randint(0, self.img_width - 1)
            y_p = random.randint(0, self.img_height - 1)
            draw.point((x_p, y_p),
                       fill=(random.randint(170, 210), random.randint(170, 210), random.randint(170, 210))) # 更淺的噪點

        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成隨機驗證碼
        code = ''.join(random.choices(CHARS[1:], k=self.code_length))  # 排除blank字符'-'

        # 生成圖片
        img = self.generate_code_image(code)

        if self.transform:
            img = self.transform(img)

        # 將驗證碼轉換為索引序列
        code_indices = [char2idx[c] for c in code]

        return img, torch.tensor(code_indices, dtype=torch.long)

def weights_init(m):
    """初始化權重"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 字符嵌入層
        self.char_emb = nn.Embedding(len(CHARS), 50)
        self.pos_emb = nn.Embedding(opt.code_length, 20)

        # 條件向量維度
        condition_dim = opt.code_length * (50 + 20)  # 5 * 70 = 350

        # 將潛在向量和條件向量映射到特徵圖
        self.fc = nn.Sequential(
            nn.Linear(opt.latent_dim + condition_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * 15 * 40),  # 對應到 (128, 15, 40)
            nn.BatchNorm1d(128 * 15 * 40),
            nn.ReLU(True)
        )

        # 轉置卷積層來上採樣 - 前半部分，輸出 (32, 60, 160)
        self.deconv_features = nn.Sequential(
            # 輸入: (128, 15, 40)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 30, 80)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 60, 160)
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # 轉置卷積層來上採樣 - 最後一層，輸出 (1, 60, 160)
        self.deconv_output = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),    # -> (1, 60, 160)
            nn.Tanh()
        )

        # 新增：分類頭 (Classification Head)
        # 輸入是 self.deconv_features 的輸出，即 (32, 60, 160)
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 30, 80)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (64, 15, 40)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 8, 20)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (128, 4, 10)
            nn.Flatten(),
            nn.Linear(128 * 4 * 10, opt.code_length * len(CHARS)) # 輸出為 (code_length * num_chars)
        )

    def forward(self, noise, code_indices):
        batch_size = noise.size(0)

        # 字符嵌入
        char_embeds = self.char_emb(code_indices)  # (batch, code_length, 50)

        # 位置嵌入
        positions = torch.arange(opt.code_length).expand(batch_size, -1)
        if cuda:
            positions = positions.cuda()
        pos_embeds = self.pos_emb(positions)  # (batch, code_length, 20)

        # 合併嵌入
        condition = torch.cat([char_embeds, pos_embeds], dim=-1)  # (batch, code_length, 70)
        condition = condition.view(batch_size, -1)  # (batch, 350)

        # 合併噪聲和條件
        gen_input = torch.cat((noise, condition), -1)  # (batch, 100 + 350)

        # 通過全連接層
        x = self.fc(gen_input)
        x = x.view(batch_size, 128, 15, 40)  # 重塑為特徵圖

        # 獲取中間特徵圖，用於分類和圖像生成
        x_features = self.deconv_features(x) # 輸出 (32, 60, 160)

        # 圖像生成
        img = self.deconv_output(x_features)

        # 字符預測
        cls_logits = self.classifier(x_features)
        # 重塑為 (batch_size, code_length, len(CHARS))
        cls_logits = cls_logits.view(batch_size, opt.code_length, len(CHARS))

        return img, cls_logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.char_embedding = nn.Embedding(len(CHARS), 50)
        self.pos_embedding = nn.Embedding(opt.code_length, 20)

        # 圖像特徵提取
        self.conv = nn.Sequential(
            # 輸入: (1, 60, 160)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 30, 80)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 15, 40)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 7, 20)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (256, 3, 10)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        condition_dim = opt.code_length * (50 + 20)  # 350
        conv_output_dim = 256 * 3 * 10  # 7680

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, code_indices):
        batch_size = img.size(0)

        # 圖像特徵提取
        img_features = self.conv(img)
        img_features = img_features.view(batch_size, -1)

        # 字符和位置嵌入
        char_embeds = self.char_embedding(code_indices)
        positions = torch.arange(opt.code_length).expand(batch_size, -1)
        if cuda:
            positions = positions.cuda()
        pos_embeds = self.pos_embedding(positions)

        condition = torch.cat([char_embeds, pos_embeds], dim=-1)
        condition = condition.view(batch_size, -1)

        # 合併圖片特徵和條件
        combined = torch.cat((img_features, condition), -1)
        validity = self.fc(combined)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()
# 新增：分類損失
classification_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# 應用權重初始化
generator.apply(weights_init)
discriminator.apply(weights_init)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    classification_loss.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def load_model(generator, discriminator, model_path):
    """載入已訓練的模型"""
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu' if not cuda else 'cuda')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        print("Model loaded successfully!")
        return True
    else:
        print(f"Model file {model_path} not found!")
        return False

def save_model(generator, discriminator, epoch, model_path):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

def generate_random_codes(batch_size, code_length):
    """生成隨機驗證碼索引"""
    codes = []
    for _ in range(batch_size):
        # CHARS[0] is '-', CHARS[1:] are actual characters
        code = [random.randint(1, len(CHARS)-1) for _ in range(code_length)]
        codes.append(code)
    return torch.tensor(codes, dtype=torch.long)

def sample_image(n_row, epoch):
    """保存生成的驗證碼圖片網格"""
    generator.eval()
    with torch.no_grad():
        # 生成噪聲
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * n_row, opt.latent_dim))))

        # 生成隨機驗證碼
        codes_indices_tensor = generate_random_codes(n_row * n_row, opt.code_length)
        if cuda:
            codes_indices_tensor = codes_indices_tensor.cuda()

        gen_imgs, _ = generator(z, codes_indices_tensor) # generator 現在返回圖片和 logits
        save_image(gen_imgs.data, f"images/epoch_{epoch}.png", nrow=n_row, normalize=True)

        # 也保存一些帶標籤的樣本
        if epoch % 10 == 0: # 每10個epoch保存帶標籤樣本
            for i in range(min(8, n_row * n_row)): # 保存最多8張
                code_str = ''.join([idx2char[idx.item()] for idx in codes_indices_tensor[i]])
                save_image(gen_imgs[i].data, f"images/sample_{epoch}_{code_str}.png", normalize=True)

    generator.train() # 設置回訓練模式

def generate_verification_codes(generator, count, output_dir, batch_size=64):
    """生成大量單一驗證碼圖片"""
    print(f"Generating {count} verification code images...")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval() # 確保生成器在評估模式
    generated_count = 0
    
    with torch.no_grad(): # 不需要計算梯度
        while generated_count < count:
            current_batch_size = min(batch_size, count - generated_count)
            
            # 生成噪聲和驗證碼
            z = FloatTensor(np.random.normal(0, 1, (current_batch_size, opt.latent_dim)))
            codes_indices_tensor = generate_random_codes(current_batch_size, opt.code_length)
            if cuda:
                codes_indices_tensor = codes_indices_tensor.cuda()
                z = z.cuda()
            
            # 生成圖片和分類預測
            gen_imgs, _ = generator(z, codes_indices_tensor) # 這裡只需要圖片，分類預測用不到
            
            # 保存圖片
            for i in range(current_batch_size):
                img_tensor = gen_imgs[i].cpu() # 轉到CPU
                code_str = ''.join([idx2char[idx.item()] for idx in codes_indices_tensor[i]])
                
                # 正確的反歸一化：從 [-1, 1] 轉回 [0, 1]
                img_normalized = (img_tensor + 1) / 2.0
                img_normalized = torch.clamp(img_normalized, 0, 1) # 確保值在 [0,1]
                
                # 轉換為PIL圖片並保存
                img_pil = transforms.ToPILImage()(img_normalized)
                
                # 檔名格式：驗證碼_序號.jpg
                filename = f"{code_str}_{generated_count:06d}.jpg"
                img_pil.save(os.path.join(output_dir, filename))
                
                generated_count += 1
                if generated_count % 1000 == 0:
                    print(f"Generated {generated_count}/{count} images")
    
    print(f"生成完成！總共生成了 {generated_count} 張驗證碼圖片")
    print(f"圖片保存在: {output_dir}")
    generator.train() # 設置回訓練模式

# 主程式邏輯
def main():
    # 如果只要載入模型並生成數據集
    if opt.generate_only:
        if opt.load_model:
            if load_model(generator, discriminator, opt.load_model):
                generate_verification_codes(generator, opt.generate_count, opt.output_dir)
            else:
                print("無法載入模型，程式結束")
        else:
            print("請使用 --load_model 指定要載入的模型路徑")
        return
    
    # 如果指定載入模型，先載入
    if opt.load_model:
        load_model(generator, discriminator, opt.load_model)
    
    # 配置數據載入器（只有在訓練時才需要）
    dataset = VerificationCodeDataset(size=opt.dataset_size, code_length=opt.code_length,
                                    img_height=opt.img_height, img_width=opt.img_width)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    # 學習率調度器
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    # ----------
    #  Training
    # ----------
    
    print("Starting training...")
    for epoch in range(opt.n_epochs):
        for i, (imgs, real_code_indices) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            # Adversarial ground truths
            valid = FloatTensor(batch_size, 1).fill_(1.0)
            fake = FloatTensor(batch_size, 1).fill_(0.0)
            
            # Configure input
            real_imgs = imgs.type(FloatTensor)
            if cuda:
                real_code_indices = real_code_indices.cuda()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and generate random codes
            z = FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
            gen_code_indices = generate_random_codes(batch_size, opt.code_length)
            if cuda:
                gen_code_indices = gen_code_indices.cuda()
                z = z.cuda()
            
            # Generate a batch of images and get classification logits
            gen_imgs, cls_logits = generator(z, gen_code_indices) # 生成器現在返回兩個值
            
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_code_indices)
            g_adv_loss = adversarial_loss(validity, valid)
            
            # 新增：分類損失
            # cls_logits 的形狀是 (batch_size, code_length, len(CHARS))
            # gen_code_indices 的形狀是 (batch_size, code_length)
            # 需要將 gen_code_indices 展開以匹配 cls_logits 的預期
            g_cls_loss = classification_loss(cls_logits.view(-1, len(CHARS)), gen_code_indices.view(-1))
            
            # 生成器總損失 = 對抗損失 + 分類損失
            g_loss = g_adv_loss + opt.lambda_cls * g_cls_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            validity_real = discriminator(real_imgs, real_code_indices)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_code_indices)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            if i % 50 == 0: # 每50個batch打印一次日誌
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f (adv: %f, cls: %f)]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), g_adv_loss.item(), g_cls_loss.item())
                )
        
        # 更新學習率
        scheduler_G.step()
        scheduler_D.step()
        
        # 每個epoch保存樣本圖片
        if epoch % 2 == 0 or epoch == opt.n_epochs - 1:
            sample_image(n_row=8, epoch=epoch)
        
        # 每10個epoch保存一次模型
        if opt.save_model and (epoch % 10 == 0 or epoch == opt.n_epochs - 1):
            model_path = f"models/cgan_epoch_{epoch}.pth"
            save_model(generator, discriminator, epoch, model_path)
    
    print("Training completed!")
    
    # 訓練完成後保存最終模型
    if opt.save_model:
        final_model_path = "models/cgan_final.pth"
        save_model(generator, discriminator, opt.n_epochs, final_model_path)
    
    # 訓練完成後根據命令行參數決定是否生成數據集
    if opt.generate_dataset:
        generate_verification_codes(generator, opt.generate_count, opt.output_dir)

if __name__ == "__main__":
    main()