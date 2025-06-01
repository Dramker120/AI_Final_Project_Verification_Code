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
from utils import plot3

# 定義字符集
CHARS = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase  # 共 63 個字符
char2idx = {ch: idx for idx, ch in enumerate(CHARS)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

os.makedirs("images", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)
os.makedirs("models", exist_ok=True)

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
parser.add_argument("--generate_only", action='store_true', help="只生成數據集，不進行訓練")
parser.add_argument("--load_model", type=str, default="", help="載入已訓練的模型路徑")
parser.add_argument("--save_model", action='store_true', help="保存訓練完成的模型")
parser.add_argument("--generate_count", type=int, default=1000, help="要生成的驗證碼圖片數量")
parser.add_argument("--output_dir", type=str, default="generated_images", help="生成圖片的輸出目錄")
parser.add_argument("--lambda_cls", type=float, default=10.0, help="Weight for G's internal classification loss")
# New hyperparameter for Generator's loss from Discriminator's classification
parser.add_argument("--lambda_gen_aux_cls", type=float, default=10.0, help="Weight for G's auxiliary classification loss (from D)")
# New hyperparameter for Discriminator's own classification loss
parser.add_argument("--lambda_disc_cls", type=float, default=10.0, help="Weight for D's own classification loss")


opt = parser.parse_args()

img_shape = (opt.channels, opt.img_height, opt.img_width)
cuda = True if torch.cuda.is_available() else False

# Original Embedding Dimensions
char_embedding_dim = 50
pos_embedding_dim = 20

class ExternalCaptchaDataset(Dataset):
    def __init__(self, root_dir='data/train_num2_var/5', img_height=60, img_width=160):
        self.root_dir = root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.samples = []

        # 掃描所有資料夾與圖片
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                label = os.path.splitext(filename)[0]
                img_path = os.path.join(root_dir, filename)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        code_indices = [char2idx[c] for c in label]
        return image, torch.tensor(code_indices, dtype=torch.long)

class VerificationCodeDataset(Dataset):
    def __init__(self, size=10000, code_length=5, img_height=60, img_width=160):
        self.size = size
        self.code_length = code_length
        self.img_height = img_height
        self.img_width = img_width
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
            try: char_w = draw.textlength(char_text, font=font)
            except AttributeError: char_w, _ = draw.textsize(char_text, font=font)
            base_x = i * char_container_width
            x_offset_in_container = max(0, (char_container_width - char_w) // 2)
            x = base_x + x_offset_in_container + random.randint(-3, 3)
            y_base = (self.img_height - font_size) // 2
            y = y_base + random.randint(-4, 4)
            y = max(0, min(y, self.img_height - font_size))
            draw.text((x, y), char_text, fill=(0,0,0), font=font)

        for _ in range(random.randint(0, 1)):
            start = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            end = (random.randint(0, self.img_width), random.randint(0, self.img_height))
            draw.line([start, end], fill=(random.randint(190, 225), random.randint(190, 225), random.randint(190, 225)), width=random.randint(1,2))
        for _ in range(random.randint(40, 80)):
            draw.point((random.randint(0, self.img_width - 1), random.randint(0, self.img_height - 1)),
                       fill=(random.randint(170, 210), random.randint(170, 210), random.randint(170, 210)))
        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        code = ''.join(random.choices(CHARS[1:], k=self.code_length))
        img = self.generate_code_image(code)
        if self.transform: img = self.transform(img)
        code_indices = [char2idx[c] for c in code]
        return img, torch.tensor(code_indices, dtype=torch.long)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.char_emb = nn.Embedding(len(CHARS), char_embedding_dim)
        self.pos_emb = nn.Embedding(opt.code_length, pos_embedding_dim)
        condition_dim = opt.code_length * (char_embedding_dim + pos_embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(opt.latent_dim + condition_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, 128 * 15 * 40), nn.BatchNorm1d(128 * 15 * 40), nn.ReLU(True))
        self.deconv_features = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), ResBlock(128),# (input: [B, 128, 15, 40])-> [B, 128, 15, 40]
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True), ResBlock(64),# [B, 64, 30, 80]
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True), ResBlock(32)) # [B, 32, 60, 160]
        self.deconv_output = nn.Sequential(nn.ConvTranspose2d(32, 1, 3, 1, 1), nn.Tanh()) # [B, 1, 60, 160]
        self.classifier = nn.Sequential( # G's internal classifier
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(2, 2), # [B, 64, 15, 40]
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(2, 2), # [B, 32, 4, 10]
            nn.Flatten(), nn.Linear(128 * 4 * 10, opt.code_length * len(CHARS)))

    def forward(self, noise, code_indices):
        batch_size = noise.size(0)
        char_embeds = self.char_emb(code_indices)
        positions = torch.arange(opt.code_length, device=noise.device).expand(batch_size, -1)
        pos_embeds = self.pos_emb(positions)
        condition = torch.cat([char_embeds, pos_embeds], dim=-1).view(batch_size, -1)
        gen_input = torch.cat((noise, condition), -1)
        x = self.fc(gen_input).view(batch_size, 128, 15, 40)
        x_features = self.deconv_features(x)
        img = self.deconv_output(x_features)
        internal_cls_logits = self.classifier(x_features)
        return img, internal_cls_logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.char_embedding = nn.Embedding(len(CHARS), char_embedding_dim)
        self.pos_embedding = nn.Embedding(opt.code_length, pos_embedding_dim)
        self.conv_features = nn.Sequential( # Shared convolutional base
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))

        conv_output_dim = 256 * 3 * 10 # After conv_features, H= (60/16 -1)*s+k ? No, it's 160/16=10, 60/16 approx 3
                                       # (H_in - F + 2P)/S + 1.  H_out = H_in / (2^4) = 160/16=10, 60/16=3.75 -> 3
                                       # So (256, 3, 10) is correct.

        # Adversarial head
        condition_dim_adv = opt.code_length * (char_embedding_dim + pos_embedding_dim)
        self.adv_head = nn.Sequential(
            nn.Linear(conv_output_dim + condition_dim_adv, 1024), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.3),
            nn.Linear(512, 1), nn.Sigmoid())

        # Classification head (operates on image features only)
        self.cls_head = nn.Sequential(
            # Input is conv_output_dim (256*3*10 = 7680)
            # Adding a bit more capacity to the classification head
            nn.Linear(conv_output_dim, 1024), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, opt.code_length * len(CHARS))
        )

    def forward(self, img, code_indices_for_adv): # code_indices are for the adversarial head
        img_conv_out = self.conv_features(img)
        img_features_flat = img_conv_out.view(img_conv_out.size(0), -1)

        # Adversarial path
        char_embeds_adv = self.char_embedding(code_indices_for_adv)
        positions_adv = torch.arange(opt.code_length, device=img.device).expand(img.size(0), -1)
        pos_embeds_adv = self.pos_embedding(positions_adv)
        condition_vec_adv = torch.cat([char_embeds_adv, pos_embeds_adv], dim=-1).view(img.size(0), -1)
        adv_input = torch.cat((img_features_flat, condition_vec_adv), -1)
        validity = self.adv_head(adv_input)

        # Classification path (on image features only)
        aux_cls_logits_flat = self.cls_head(img_features_flat) # Pass only image features
        aux_cls_logits = aux_cls_logits_flat.view(img.size(0), opt.code_length, len(CHARS))

        return validity, aux_cls_logits

adversarial_loss_fn = torch.nn.BCELoss()
classification_loss_fn = torch.nn.CrossEntropyLoss()

generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init); discriminator.apply(weights_init)

if cuda:
    generator.cuda(); discriminator.cuda()
    adversarial_loss_fn.cuda(); classification_loss_fn.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def load_model_func(generator_instance, discriminator_instance, model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        loc = 'cuda' if cuda else 'cpu'
        checkpoint = torch.load(model_path, map_location=loc)
        generator_instance.load_state_dict(checkpoint['generator'])
        discriminator_instance.load_state_dict(checkpoint['discriminator'])
        print("Model loaded successfully!")
        return True
    print(f"Model file {model_path} not found!"); return False

def save_model_func(generator_instance, discriminator_instance, epoch_num, model_path):
    torch.save({
        'epoch': epoch_num,
        'generator': generator_instance.state_dict(),
        'discriminator': discriminator_instance.state_dict(),
    }, model_path); print(f"Model saved to {model_path}")

def generate_random_codes_tensor(batch_size, code_len):
    codes = [[random.randint(1, len(CHARS)-1) for _ in range(code_len)] for _ in range(batch_size)]
    return torch.tensor(codes, dtype=torch.long)

def sample_image_func(n_row, epoch_num):
    generator.eval()
    with torch.no_grad():
        z = FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim)))
        gen_codes_idx = generate_random_codes_tensor(n_row**2, opt.code_length).type(LongTensor)
        gen_imgs, _ = generator(z, gen_codes_idx) # G internal logits ignored here
        save_image(gen_imgs.data, f"images/epoch_{epoch_num}.png", nrow=n_row, normalize=True)
        if epoch_num % 10 == 0:
            for i in range(min(8, n_row**2)):
                code_str = ''.join([idx2char[idx.item()] for idx in gen_codes_idx[i]])
                save_image(gen_imgs[i].data, f"images/sample_{epoch_num}_{code_str}.png", normalize=True)
    generator.train()

def generate_verification_codes_func(gen, count, out_dir, batch_sz):
    print(f"Generating {count} images..."); os.makedirs(out_dir, exist_ok=True)
    gen.eval(); generated_count = 0
    with torch.no_grad():
        while generated_count < count:
            curr_batch_sz = min(batch_sz, count - generated_count)
            z_noise = FloatTensor(np.random.normal(0, 1, (curr_batch_sz, opt.latent_dim)))
            codes_idx = generate_random_codes_tensor(curr_batch_sz, opt.code_length).type(LongTensor)
            # gen_imgs, _ = gen(z_noise.cuda() if cuda else z_noise, codes_idx.cuda() if cuda else codes_idx)
            z_noise = z_noise.cuda() if cuda else z_noise
            codes_idx = codes_idx.cuda() if cuda else codes_idx
            gen_imgs, _ = gen(z_noise, codes_idx)


            for i in range(curr_batch_sz):
                img_tensor = (gen_imgs[i].cpu().data + 1) / 2.0 # Denormalize from [-1,1] to [0,1]
                img_tensor.clamp_(0,1)
                code_s = ''.join([idx2char[idx.item()] for idx in codes_idx[i]])
                # Add unique identifier to filename to prevent overwrites if codes repeat
                # and batch generation leads to multiple files for the same code in one go.
                fname = f"{code_s}.jpg"
                transforms.ToPILImage()(img_tensor).save(os.path.join(out_dir, fname))
            generated_count += curr_batch_sz
            if generated_count % 1000 == 0 or generated_count == count : print(f"Generated {generated_count}/{count} images")
    print(f"Generation complete! {generated_count} images saved to {out_dir}"); gen.train()

def main():
    print(opt)
    if opt.generate_only:
        if opt.load_model:
            if load_model_func(generator, discriminator, opt.load_model):
                generate_verification_codes_func(generator, opt.generate_count, opt.output_dir, opt.batch_size)
            else: print("Cannot load model, exiting.")
        else: print("Please use --load_model to specify a model path for generation.")
        return

    if opt.load_model: load_model_func(generator, discriminator, opt.load_model)

    # dataset = ExternalCaptchaDataset(root_dir=f"data/train_num2_var/{opt.code_length}", img_height=opt.img_height, img_width=opt.img_width)
    dataset = VerificationCodeDataset(opt.dataset_size, opt.code_length, opt.img_height, opt.img_width) 
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    print("Starting cGAN training...")

    g_loss_adversarial = []
    g_loss_external_classes = []
    g_loss_internal_classes = []
    g_total_loss = []
    for epoch in range(opt.n_epochs):
        for i, (real_imgs, real_code_indices) in enumerate(dataloader):
            batch_size_actual = real_imgs.shape[0]
            valid_labels = FloatTensor(batch_size_actual, 1).fill_(1.0)
            fake_labels = FloatTensor(batch_size_actual, 1).fill_(0.0)
            real_imgs = real_imgs.type(FloatTensor)
            real_code_indices = real_code_indices.type(LongTensor)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            z_noise = FloatTensor(np.random.normal(0, 1, (batch_size_actual, opt.latent_dim)))
            # Target codes for G to generate
            gen_target_code_indices = generate_random_codes_tensor(batch_size_actual, opt.code_length).type(LongTensor)

            gen_imgs, g_internal_cls_logits = generator(z_noise, gen_target_code_indices)
            
            # Pass generated images through D
            # For G's loss, D needs to process gen_imgs with gen_target_code_indices as the adv condition
            d_adv_pred_on_fake, d_aux_cls_pred_on_fake = discriminator(gen_imgs, gen_target_code_indices)

            g_loss_adv = adversarial_loss_fn(d_adv_pred_on_fake, valid_labels) # G wants D to think fake is valid
            g_loss_internal_cls = classification_loss_fn(
                g_internal_cls_logits.reshape(-1, len(CHARS)),
                gen_target_code_indices.reshape(-1)
            )
            g_loss_external_cls = classification_loss_fn( # G wants D to classify fake images correctly
                d_aux_cls_pred_on_fake.reshape(-1, len(CHARS)),
                gen_target_code_indices.reshape(-1)
            )
            g_loss = g_loss_adv + \
                     opt.lambda_cls * g_loss_internal_cls + \
                     opt.lambda_gen_aux_cls * g_loss_external_cls
            g_loss.backward()
            optimizer_G.step()

            if i % 5 == 0:
                # --- Train Discriminator ---
                optimizer_D.zero_grad()
                
                # D's loss on real images
                d_adv_pred_real, d_aux_cls_pred_real = discriminator(real_imgs, real_code_indices)
                d_loss_real_adv = adversarial_loss_fn(d_adv_pred_real, valid_labels)
                d_loss_real_cls = classification_loss_fn(
                    d_aux_cls_pred_real.reshape(-1, len(CHARS)),
                    real_code_indices.reshape(-1)
                )

                # D's loss on fake images
                # For D's loss, D needs to process gen_imgs.detach() with gen_target_code_indices as the adv condition
                d_adv_pred_fake, d_aux_cls_pred_fake = discriminator(gen_imgs.detach(), gen_target_code_indices)
                d_loss_fake_adv = adversarial_loss_fn(d_adv_pred_fake, fake_labels)
                d_loss_fake_cls = classification_loss_fn( # D learns to classify even the fake ones (based on G's attempt)
                    d_aux_cls_pred_fake.reshape(-1, len(CHARS)),
                    gen_target_code_indices.reshape(-1)
                )
                
                d_loss_adv_total = (d_loss_real_adv + d_loss_fake_adv) / 2
                d_loss_cls_total = d_loss_real_cls + d_loss_fake_cls # Sum of cls losses for D
                
                d_loss = d_loss_adv_total + opt.lambda_disc_cls * d_loss_cls_total
                d_loss.backward()
                optimizer_D.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D adv: {d_loss_adv_total.item():.4f}, D cls: {d_loss_cls_total.item():.4f}] "
                      f"[G adv: {g_loss_adv.item():.4f}, G int_cls: {g_loss_internal_cls.item():.4f}, G ext_cls: {g_loss_external_cls.item():.4f}] "
                      f"[LR: {optimizer_G.param_groups[0]['lr']:.1e}]")
            if i==150:
                g_loss_adversarial.append(g_loss_adv.item())
                g_loss_internal_classes.append(g_loss_internal_cls.item())
                g_loss_external_classes.append(g_loss_external_cls.item())
                g_total_loss.append(g_loss.item())
        scheduler_G.step(); scheduler_D.step()
        if epoch % 2 == 0 or epoch == opt.n_epochs - 1: sample_image_func(8, epoch)
        if opt.save_model and (epoch > 0 and epoch % 10 == 0 or epoch == opt.n_epochs - 1):
            save_model_func(generator, discriminator, epoch, f"models/cgan_epoch_{epoch}.pth")

    print("Training completed!")
    plot3(g_loss_adversarial,g_loss_internal_classes,g_loss_external_classes,g_total_loss)
    if opt.save_model: save_model_func(generator, discriminator, opt.n_epochs, "models/cgan_final.pth")
    if opt.generate_dataset: generate_verification_codes_func(generator, opt.generate_count, opt.output_dir, opt.batch_size)

if __name__ == "__main__":
    main()