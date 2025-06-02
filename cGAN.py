import argparse
import os
import numpy as np
import math
import string
import random
import shutil 

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets # Not used in the provided cGAN.py
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import plot3 # Removed as utils.py is not provided

# --- Global Definitions ---
CHARS = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase  # 共 63 個字符
char2idx = {ch: idx for idx, ch in enumerate(CHARS)} #
idx2char = {idx: ch for ch, idx in char2idx.items()} #

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Original Embedding Dimensions (will be used directly by models)
char_embedding_dim = 50 #
pos_embedding_dim = 20  #

# --- Dataset Definitions ---
class ExternalCaptchaDataset(Dataset): # Based on user's cGAN.py
    def __init__(self, root_dir, current_code_length, img_height=60, img_width=160): # Added current_code_length
        self.root_dir = root_dir
        self.code_length = current_code_length # Store current_code_length
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.samples = []
        if not os.path.isdir(root_dir):
            print(f"Warning: Dataset directory not found for length {self.code_length}: {root_dir}")
            return

        for filename in os.listdir(root_dir): #
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')): # Allow more extensions
                label_from_filename = os.path.splitext(filename)[0]
                # Handle potential suffixes like _timestamp if they exist, assuming label is first part
                label_parts = label_from_filename.split('_')
                label = label_parts[0]

                if len(label) == self.code_length and all(c in char2idx for c in label):
                    img_path = os.path.join(root_dir, filename)
                    self.samples.append((img_path, label))
        
        if self.samples:
            print(f"Loaded {len(self.samples)} samples from {root_dir} for code length {self.code_length}")
        else:
            print(f"Warning: No valid samples found in {root_dir} for code_length={self.code_length}.")

    def __len__(self):
        return len(self.samples) #

    def __getitem__(self, idx): #
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB') #
            if self.transform: image = self.transform(image) #
            code_indices = [char2idx[c] for c in label] #
            return image, torch.tensor(code_indices, dtype=torch.long) #
        except Exception as e:
            print(f"Error loading image {img_path} with label {label}: {e}")
            if idx + 1 < len(self.samples): return self.__getitem__(idx + 1)
            dummy_img = torch.zeros((1, self.img_height, self.img_width), dtype=torch.float)
            dummy_label = torch.zeros((self.code_length,), dtype=torch.long)
            return dummy_img, dummy_label

class VerificationCodeDataset(Dataset): # From user's cGAN.py
    def __init__(self, size, current_code_length, img_height, img_width): # Changed signature
        self.size = size #
        self.code_length = current_code_length # Use passed arg
        self.img_height = img_height #
        self.img_width = img_width   #
        self.transform = transforms.Compose([ #
            transforms.Grayscale(num_output_channels=1), #
            transforms.ToTensor(), #
            transforms.Normalize([0.5], [0.5]) #
        ])

    def generate_code_image(self, code): # (Content identical to user's file)
        img = Image.new('RGB', (self.img_width, self.img_height), 'white')
        draw = ImageDraw.Draw(img)
        font_size = 40
        try: font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try: font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except IOError:
                try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except IOError: font = ImageFont.load_default()
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
    def __len__(self): return self.size #
    def __getitem__(self, idx): #
        code = ''.join(random.choices(CHARS[1:], k=self.code_length)) #
        img = self.generate_code_image(code) #
        if self.transform: img = self.transform(img) #
        code_indices = [char2idx[c] for c in code] #
        return img, torch.tensor(code_indices, dtype=torch.long) #

# --- Model Definitions ---
def weights_init(m): #
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ResBlock(nn.Module): #
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return F.relu(x + self.block(x)) #

class Generator(nn.Module): # Based on user's cGAN.py
    def __init__(self, opt_for_model): # Takes opt to access opt.code_length, opt.latent_dim
        super(Generator, self).__init__()
        self.opt = opt_for_model # Store opt for internal use
        self.char_emb = nn.Embedding(len(CHARS), char_embedding_dim) #
        self.pos_emb = nn.Embedding(self.opt.code_length, pos_embedding_dim) # Uses opt.code_length
        condition_dim = self.opt.code_length * (char_embedding_dim + pos_embedding_dim) #

        self.fc = nn.Sequential( #
            nn.Linear(self.opt.latent_dim + condition_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(True), #
            nn.Linear(1024, 128 * 15 * 40), nn.BatchNorm1d(128 * 15 * 40), nn.ReLU(True)) #
        self.deconv_features = nn.Sequential( #
            nn.ConvTranspose2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), ResBlock(128), #
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True), ResBlock(64), #
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True), ResBlock(32)) #
        self.deconv_output = nn.Sequential(nn.ConvTranspose2d(32, 1, 3, 1, 1), nn.Tanh()) #
        self.classifier = nn.Sequential( #
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(2, 2), #
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(2, 2), #
            nn.Flatten(), nn.Linear(128 * 4 * 10, self.opt.code_length * len(CHARS))) #

    def forward(self, noise, code_indices): #
        batch_size = noise.size(0)
        char_embeds = self.char_emb(code_indices) #
        positions = torch.arange(self.opt.code_length, device=noise.device).expand(batch_size, -1) #
        pos_embeds = self.pos_emb(positions) #
        condition = torch.cat([char_embeds, pos_embeds], dim=-1).view(batch_size, -1) #
        gen_input = torch.cat((noise, condition), -1) #
        x = self.fc(gen_input).view(batch_size, 128, 15, 40) #
        x_features = self.deconv_features(x) #
        img = self.deconv_output(x_features) #
        internal_cls_logits = self.classifier(x_features).view(batch_size, self.opt.code_length, len(CHARS)) # Ensure reshape
        return img, internal_cls_logits #

class Discriminator(nn.Module): # Based on user's cGAN.py
    def __init__(self, opt_for_model): # Takes opt
        super(Discriminator, self).__init__()
        self.opt = opt_for_model # Store opt
        self.char_embedding = nn.Embedding(len(CHARS), char_embedding_dim) #
        self.pos_embedding = nn.Embedding(self.opt.code_length, pos_embedding_dim) # Uses opt.code_length
        self.conv_features = nn.Sequential( #
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True), #
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True), #
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), #
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True)) #
        conv_output_dim = 256 * 3 * 10 #
        condition_dim_adv = self.opt.code_length * (char_embedding_dim + pos_embedding_dim) #
        self.adv_head = nn.Sequential( #
            nn.Linear(conv_output_dim + condition_dim_adv, 1024), nn.LeakyReLU(0.2, True), nn.Dropout(0.3), #
            nn.Linear(1024, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.3), #
            nn.Linear(512, 1), nn.Sigmoid()) #
        self.cls_head = nn.Sequential( #
            nn.Linear(conv_output_dim, 1024), nn.ReLU(True), nn.Dropout(0.3), #
            nn.Linear(1024, 512), nn.ReLU(True), nn.Dropout(0.3), #
            nn.Linear(512, self.opt.code_length * len(CHARS))) #

    def forward(self, img, code_indices_for_adv): #
        img_conv_out = self.conv_features(img) #
        img_features_flat = img_conv_out.view(img_conv_out.size(0), -1) #
        char_embeds_adv = self.char_embedding(code_indices_for_adv) #
        positions_adv = torch.arange(self.opt.code_length, device=img.device).expand(img.size(0), -1) #
        pos_embeds_adv = self.pos_embedding(positions_adv) #
        condition_vec_adv = torch.cat([char_embeds_adv, pos_embeds_adv], dim=-1).view(img.size(0), -1) #
        adv_input = torch.cat((img_features_flat, condition_vec_adv), -1) #
        validity = self.adv_head(adv_input) #
        aux_cls_logits_flat = self.cls_head(img_features_flat) #
        aux_cls_logits = aux_cls_logits_flat.view(img.size(0), self.opt.code_length, len(CHARS)) #
        return validity, aux_cls_logits #

# --- Helper Functions (largely from user's cGAN.py) ---
def load_model_func(generator_instance, discriminator_instance, model_path): #
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        loc = 'cuda' if cuda else 'cpu'
        try:
            checkpoint = torch.load(model_path, map_location=loc) #
            generator_instance.load_state_dict(checkpoint['generator']) #
            if discriminator_instance and 'discriminator' in checkpoint: #
                 discriminator_instance.load_state_dict(checkpoint['discriminator']) #
            print(f"Model loaded successfully from {model_path}!")
            return True
        except Exception as e:
            print(f"Error loading model {model_path}: {e}. Might be due to structure mismatch or missing keys.")
            return False
    print(f"Model file {model_path} not found!"); return False

def save_model_func(generator_instance, discriminator_instance, epoch_num, model_path): #
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({ #
        'epoch': epoch_num, #
        'generator': generator_instance.state_dict(), #
        'discriminator': discriminator_instance.state_dict(), #
    }, model_path); print(f"Model saved to {model_path}")

def generate_random_codes_tensor(batch_size, current_code_length_for_gen): # (Added current_code_length_for_gen)
    codes = [[random.randint(1, len(CHARS)-1) for _ in range(current_code_length_for_gen)] for _ in range(batch_size)] #
    return torch.tensor(codes, dtype=torch.long).type(LongTensor) # Ensure on correct device

def sample_image_during_train(generator_instance, n_row, epoch_num, opt_for_sampling): # Renamed and adapted
    generator_instance.eval() #
    with torch.no_grad(): #
        z = FloatTensor(np.random.normal(0, 1, (n_row**2, opt_for_sampling.latent_dim))) #
        # Use current opt_for_sampling.code_length
        gen_codes_idx = generate_random_codes_tensor(n_row**2, opt_for_sampling.code_length).type(LongTensor) #
        gen_imgs, _ = generator_instance(z, gen_codes_idx) #
        
        image_dir_epoch_samples = os.path.join(opt_for_sampling.base_image_dir_train_samples, f"{opt_for_sampling.code_length}", "epochs")
        image_dir_labeled_samples = os.path.join(opt_for_sampling.base_image_dir_train_samples, f"{opt_for_sampling.code_length}", "labeled")
        os.makedirs(image_dir_epoch_samples, exist_ok=True)
        os.makedirs(image_dir_labeled_samples, exist_ok=True)
        
        save_image(gen_imgs.data, os.path.join(image_dir_epoch_samples, f"epoch_{epoch_num}.png"), nrow=n_row, normalize=True) #
        if epoch_num % 10 == 0: #
            for i in range(min(8, n_row**2)): #
                code_str = ''.join([idx2char[idx.item()] for idx in gen_codes_idx[i]]) #
                save_image(gen_imgs[i].data, os.path.join(image_dir_labeled_samples, f"sample_e{epoch_num}_{code_str}.png"), normalize=True) #
    generator_instance.train() #

def generate_final_images_after_training(
    trained_generator, opt_for_generation): # Adapted from user's generate_verification_codes_func

    output_dir_final = opt_for_generation.output_dir_template_final_gen.format(length=opt_for_generation.code_length)
    gen_count = opt_for_generation.generate_count_final # Use specific count for this phase
    
    print(f"Generating {gen_count} final images for length {opt_for_generation.code_length} into {output_dir_final}...")
    os.makedirs(output_dir_final, exist_ok=True)
    
    trained_generator.eval() #
    generated_image_count = 0 # Renamed from generated_count to avoid conflict
    with torch.no_grad(): #
        while generated_image_count < gen_count: #
            curr_batch_sz = min(opt_for_generation.batch_size_final_gen, gen_count - generated_image_count) #
            if curr_batch_sz <= 0: break

            z_noise = FloatTensor(np.random.normal(0, 1, (curr_batch_sz, opt_for_generation.latent_dim))) #
            # Use current opt_for_generation.code_length
            codes_idx = generate_random_codes_tensor(curr_batch_sz, opt_for_generation.code_length).type(LongTensor) #
            
            gen_imgs, _ = trained_generator(z_noise, codes_idx) #
            
            for i in range(curr_batch_sz): #
                if generated_image_count >= gen_count: break
                img_tensor = (gen_imgs[i].cpu().data + 1) / 2.0 #
                img_tensor.clamp_(0,1) #
                code_s = ''.join([idx2char[idx.item()] for idx in codes_idx[i]]) #
                
                fname = f"{code_s}.png" # Filename is the content + .png (using .png as requested)
                
                # Handle potential filename collisions if strict "10000 files" is needed and codes repeat.
                # For now, if code_s.png is generated multiple times, it will be overwritten.
                # If 10000 *unique file entries* are needed even if content repeats, 
                # a suffix like _{generated_image_count} would be required.
                # Based on "檔名為該驗證碼的內容", we assume overwriting identical codes is acceptable if generated multiple times.
                
                pil_img = transforms.ToPILImage()(img_tensor) #
                # Ensure image is saved at the correct dimensions (though G should output correctly)
                if pil_img.size != (opt_for_generation.img_width, opt_for_generation.img_height):
                   pil_img = pil_img.resize((opt_for_generation.img_width, opt_for_generation.img_height), Image.Resampling.LANCZOS)
                
                pil_img.save(os.path.join(output_dir_final, fname)) #
                generated_image_count += 1 #

            if generated_image_count % 1000 == 0 or generated_image_count >= gen_count : #
                print(f"Length {opt_for_generation.code_length}: Generated {generated_image_count}/{gen_count} images")

    print(f"Final image generation complete for length {opt_for_generation.code_length}! {generated_image_count} images saved to {output_dir_final}")
    # No need to set trained_generator.train() here as this is the end of its use for this length


# --- Training and Generation Function for a single length ---
def train_and_generate_for_single_length(opt_current_run):
    # This 'opt_current_run' will be the global 'opt' for models within this function's scope.
    # We need to ensure that Generator and Discriminator use this specific opt.
    # One way is to pass it to their __init__ or rely on the global 'opt' being set correctly before model instantiation.
    # The simplest way to match the original structure is to temporarily set the global `opt`
    # or ensure models are defined to take `opt` as an argument.
    # Given the original models `Generator()` and `Discriminator()` directly use the global `opt`,
    # we will modify the global `opt` for the duration of this function call for a specific length.
    
    # The models Generator() and Discriminator() in the user's file directly use the global `opt`
    # so, we ensure the global `opt` (which is `opt_current_run` here) has the correct `code_length`
    # before instantiating them.

    print(f"\n===== Processing Code Length: {opt_current_run.code_length} =====")
    print(f"Using effective opt for this run: {opt_current_run}")
    
    # --- Initialize models for the current code_length ---
    # Models will use the current `opt_current_run.code_length` via their __init__ if they access `opt.code_length`
    current_generator = Generator(opt_current_run) # Pass the current opt object
    current_discriminator = Discriminator(opt_current_run) # Pass the current opt object

    current_generator.apply(weights_init) #
    current_discriminator.apply(weights_init) #

    if cuda:
        current_generator.cuda() #
        current_discriminator.cuda() #

    # --- Load pre-trained model if specified ---
    if opt_current_run.load_model_path_template: # (logic adapted from original main)
        model_load_path = opt_current_run.load_model_path_template.format(length=opt_current_run.code_length)
        if os.path.exists(model_load_path):
            print(f"Attempting to load pre-trained model for length {opt_current_run.code_length} from: {model_load_path}")
            load_model_func(current_generator, current_discriminator, model_load_path)
        # else: # Implicitly trains from scratch if not found, matching original flow
            # print(f"No pre-trained model found at {model_load_path}. Training from scratch for length {opt_current_run.code_length}.")
    # elif opt_current_run.load_model: # This was from the original single-model load logic
        # load_model_func(current_generator, current_discriminator, opt_current_run.load_model)


    # --- Dataset ---
    # The original script used VerificationCodeDataset directly in main.
    # We will allow selection or default to VerificationCodeDataset as in the original main.
    dataset_path_for_external = opt_current_run.base_train_dataset_path_template.format(length=opt_current_run.code_length)
    
    if opt_current_run.use_external_dataset_train:
        print(f"Using ExternalCaptchaDataset for code length {opt_current_run.code_length} from: {dataset_path_for_external}")
        dataset = ExternalCaptchaDataset(root_dir=dataset_path_for_external, 
                                         current_code_length=opt_current_run.code_length, 
                                         img_height=opt_current_run.img_height, 
                                         img_width=opt_current_run.img_width)
    else: # Default to VerificationCodeDataset as in the original main's active line
        print(f"Using dynamically generated VerificationCodeDataset for length {opt_current_run.code_length} (size: {opt_current_run.dataset_size_dynamic_train})")
        dataset = VerificationCodeDataset(size=opt_current_run.dataset_size_dynamic_train, 
                                          current_code_length=opt_current_run.code_length, # Pass current length
                                          img_height=opt_current_run.img_height, 
                                          img_width=opt_current_run.img_width)


    if len(dataset) == 0:
        print(f"ERROR: Dataset is empty for length {opt_current_run.code_length}. Skipping this length.")
        return

    dataloader = DataLoader(dataset, opt_current_run.batch_size, shuffle=True, num_workers=opt_current_run.n_cpu, drop_last=True, pin_memory=cuda) #

    # --- Optimizers and Schedulers ---
    optimizer_G = torch.optim.Adam(current_generator.parameters(), lr=opt_current_run.lr, betas=(opt_current_run.b1, opt_current_run.b2)) #
    optimizer_D = torch.optim.Adam(current_discriminator.parameters(), lr=opt_current_run.lr, betas=(opt_current_run.b1, opt_current_run.b2)) #
    
    # Scheduler from user's cGAN.py
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5) #
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5) #

    # --- Loss Functions ---
    adversarial_loss_fn = torch.nn.BCELoss() #
    classification_loss_fn = torch.nn.CrossEntropyLoss() #
    if cuda:
        adversarial_loss_fn.cuda() #
        classification_loss_fn.cuda() #
    
    print(f"Starting training for {opt_current_run.n_epochs} epochs with {len(dataset)} images (Length: {opt_current_run.code_length})...")
    
    # Loss tracking lists (from user's cGAN.py)
    g_loss_adversarial_list = []
    g_loss_external_classes_list = []
    g_loss_internal_classes_list = []
    g_total_loss_list = []

    for epoch in range(opt_current_run.n_epochs): #
        for i, (real_imgs, real_code_indices) in enumerate(dataloader): #
            batch_size_actual = real_imgs.shape[0] #
            if batch_size_actual == 0: continue

            valid_labels = FloatTensor(batch_size_actual, 1).fill_(1.0) #
            # Using smoothed fake labels can sometimes help stabilize training.
            # fake_labels = FloatTensor(batch_size_actual, 1).fill_(0.0) 
            fake_labels = FloatTensor(batch_size_actual, 1).uniform_(0.0, 0.2) # Smoothed slightly


            real_imgs = real_imgs.type(FloatTensor) #
            real_code_indices = real_code_indices.type(LongTensor) #

            # --- Train Generator ---
            optimizer_G.zero_grad() #
            z_noise = FloatTensor(np.random.normal(0, 1, (batch_size_actual, opt_current_run.latent_dim))) #
            # Target codes for G to generate, using current opt_current_run.code_length
            gen_target_code_indices = generate_random_codes_tensor(batch_size_actual, opt_current_run.code_length) #

            gen_imgs, g_internal_cls_logits = current_generator(z_noise, gen_target_code_indices) #
            
            d_adv_pred_on_fake, d_aux_cls_pred_on_fake = current_discriminator(gen_imgs, gen_target_code_indices) #

            g_loss_adv = adversarial_loss_fn(d_adv_pred_on_fake, valid_labels) #
            g_loss_internal_cls = classification_loss_fn( #
                g_internal_cls_logits.reshape(-1, len(CHARS)), #
                gen_target_code_indices.reshape(-1) #
            )
            g_loss_external_cls = classification_loss_fn( #
                d_aux_cls_pred_on_fake.reshape(-1, len(CHARS)), #
                gen_target_code_indices.reshape(-1) #
            )
            g_loss = g_loss_adv + \
                     opt_current_run.lambda_cls * g_loss_internal_cls + \
                     opt_current_run.lambda_gen_aux_cls * g_loss_external_cls #
            
            if not (torch.isnan(g_loss) or torch.isinf(g_loss)): #
                g_loss.backward() #
                optimizer_G.step() #
            
            # --- Train Discriminator (conditionally) ---
            d_loss_adv_total_batch_val = -1.0 # for logging if D not updated
            d_loss_cls_total_batch_val = -1.0 # for logging

            if i % opt_current_run.d_updates_interval == 0: #
                optimizer_D.zero_grad() #
                d_adv_pred_real, d_aux_cls_pred_real = current_discriminator(real_imgs, real_code_indices) #
                d_loss_real_adv = adversarial_loss_fn(d_adv_pred_real, valid_labels) #
                d_loss_real_cls = classification_loss_fn( #
                    d_aux_cls_pred_real.reshape(-1, len(CHARS)), #
                    real_code_indices.reshape(-1) #
                )
                d_adv_pred_fake, d_aux_cls_pred_fake = current_discriminator(gen_imgs.detach(), gen_target_code_indices) #
                d_loss_fake_adv = adversarial_loss_fn(d_adv_pred_fake, fake_labels) #
                d_loss_fake_cls = classification_loss_fn( #
                    d_aux_cls_pred_fake.reshape(-1, len(CHARS)), #
                    gen_target_code_indices.reshape(-1) #
                )
                d_loss_adv_total_batch = (d_loss_real_adv + d_loss_fake_adv) / 2 #
                d_loss_cls_total_batch = d_loss_real_cls + d_loss_fake_cls #
                d_loss = d_loss_adv_total_batch + opt_current_run.lambda_disc_cls * d_loss_cls_total_batch #
                
                if not (torch.isnan(d_loss) or torch.isinf(d_loss)): #
                    d_loss.backward() #
                    optimizer_D.step() #
                d_loss_adv_total_batch_val = d_loss_adv_total_batch.item()
                d_loss_cls_total_batch_val = d_loss_cls_total_batch.item()


            if i % opt_current_run.log_interval_batch == 0: #
                print(f"[Epoch:{epoch}/{opt_current_run.n_epochs} Batch:{i}/{len(dataloader)}] " 
                      f"[D adv: {d_loss_adv_total_batch_val:.4f}, D cls: {d_loss_cls_total_batch_val:.4f}] " 
                      f"[G adv: {g_loss_adv.item():.4f}, G int_cls: {g_loss_internal_cls.item():.4f}, G ext_cls: {g_loss_external_cls.item():.4f}] " 
                      f"[LR D:{optimizer_D.param_groups[0]['lr']:.1e} G:{optimizer_G.param_groups[0]['lr']:.1e}]")
            
            # Removed plotting data collection
            if i==150:
                g_loss_adversarial_list.append(g_loss_adv.item())
                g_loss_internal_classes_list.append(g_loss_internal_cls.item())
                g_loss_external_classes_list.append(g_loss_external_cls.item())
                g_total_loss_list.append(g_loss.item())

        scheduler_G.step(); scheduler_D.step() #
        
        if epoch % opt_current_run.sample_image_epoch_interval == 0 or epoch == opt_current_run.n_epochs - 1: # (logic adapted)
             sample_image_during_train(current_generator, 8, epoch, opt_current_run) # Pass current opt
        
        if opt_current_run.save_model_epoch_interval > 0 and \
           (epoch > 0 and epoch % opt_current_run.save_model_epoch_interval == 0 or epoch == opt_current_run.n_epochs - 1): # (logic adapted)
            model_save_path_epoch = os.path.join(opt_current_run.model_save_dir, f"cgan_len{opt_current_run.code_length}_epoch_{epoch}.pth") # (path adapted)
            save_model_func(current_generator, current_discriminator, epoch, model_save_path_epoch) #

    print(f"Training completed for code length {opt_current_run.code_length}!") #
    final_model_path = os.path.join(opt_current_run.model_save_dir, f"cgan_len{opt_current_run.code_length}_final.pth")
    save_model_func(current_generator, current_discriminator, opt_current_run.n_epochs, final_model_path) # (path adapted)
    
    plot3(g_loss_adversarial_list,g_loss_internal_classes_list,g_loss_external_classes_list,g_total_loss_list, opt_current_run.code_length)
    
    # --- Generate final images after training this length ---
    print(f"\n----- Generating {opt_current_run.generate_count_final} images for code length {opt_current_run.code_length} after training -----")
    generate_final_images_after_training(
        trained_generator=current_generator,
        opt_for_generation=opt_current_run # Pass the full opt for this run
    )


# --- Main Execution Block for Combined Workflow ---
if __name__ == "__main__":
    # This parser is for the overall script controlling multiple runs
    master_parser = argparse.ArgumentParser(description="Combined cGAN Trainer and Final Image Generator")
    
    # Workflow controls
    master_parser.add_argument("--code_lengths_to_process", type=str, default="4,5,6,7", 
                               help="Comma-separated list of code lengths to train and generate for.")
    master_parser.add_argument("--base_train_dataset_path_template", type=str, default="data/train_num2_var/{length}", 
                               help="Training dataset path template. {length} is replaced by code_length.")
    master_parser.add_argument("--use_external_dataset_train", action='store_true', 
                               help="Use ExternalCaptchaDataset. If false, uses VerificationCodeDataset (dynamic).")
    master_parser.add_argument("--dataset_size_dynamic_train", type=int, default=10000, # From original opt.dataset_size
                               help="Size for dynamic VerificationCodeDataset if used.")
    
    # Training parameters (will be part of the opt object passed to train_and_generate)
    # These mirror the original script's opt arguments
    master_parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs for each length.")
    master_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    master_parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    master_parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1.")
    master_parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2.")
    master_parser.add_argument("--n_cpu", type=int, default=8, help="N CPU for DataLoader.")
    master_parser.add_argument("--latent_dim", type=int, default=100, help="Latent dim.")
    master_parser.add_argument("--img_height", type=int, default=60, help="Image height.")
    master_parser.add_argument("--img_width", type=int, default=160, help="Image width.")
    # opt.channels is implicitly 1 due to Grayscale transform. Not needed as direct model param.

    master_parser.add_argument("--lambda_cls", type=float, default=10.0, help="Weight for G's internal cls loss.") #
    master_parser.add_argument("--lambda_gen_aux_cls", type=float, default=10.0, help="Weight for G's external cls loss (from D).") #
    master_parser.add_argument("--lambda_disc_cls", type=float, default=10.0, help="Weight for D's own cls loss.") #
    master_parser.add_argument("--d_updates_interval", type=int, default=5, help="Update D every N G updates.") #
    
    master_parser.add_argument("--log_interval_batch", type=int, default=50, help="Log training progress every N batches.") # (used 50)
    master_parser.add_argument("--sample_image_epoch_interval", type=int, default=2, help="Save sample images every N epochs.") #
    master_parser.add_argument("--save_model_epoch_interval", type=int, default=10, help="Save model checkpoint every N epochs (0 to disable).") #
    master_parser.add_argument("--model_save_dir", type=str, default="models", 
                               help="Base directory to save all trained models.") # Original was "models"
    master_parser.add_argument("--base_image_dir_train_samples", type=str, default="training_images", 
                               help="Base directory for saving sample images during training.") # Original was "images"
    master_parser.add_argument("--load_model_path_template", type=str, default="", 
                               help="Optional: Path template to load pre-trained models (e.g., models/combined_output/cgan_len{length}_final.pth).")

    # Generation parameters
    master_parser.add_argument("--generate_count_final", type=int, default=10000, 
                               help="Number of final images to generate for each code length.")
    master_parser.add_argument("--output_dir_template_final_gen", type=str, default="data/dataset3_images/{length}", 
                               help="Output directory template for final generated images. {length} is replaced.")
    master_parser.add_argument("--batch_size_final_gen", type=int, default=128, # A reasonable default for generation
                               help="Batch size for final image generation phase.")

    # Parse arguments once for the master script
    # The 'opt' object created here will be modified for each length if necessary,
    # or its relevant fields passed.
    # In this version, we create a new opt namespace for each run to avoid global opt issues.
    master_opt_namespace = master_parser.parse_args()
    
    print("===== Master Configuration (defaults for each run unless overridden by loop) =====")
    for k, v in vars(master_opt_namespace).items():
        print(f"{k}: {v}")
    print("==================================================================================")

    # Create base directories if they don't exist from master_opt
    os.makedirs(master_opt_namespace.model_save_dir, exist_ok=True)
    os.makedirs(master_opt_namespace.base_image_dir_train_samples, exist_ok=True)
    # Final generation output directories will be created per length.

    code_lengths_to_run = [int(x.strip()) for x in master_opt_namespace.code_lengths_to_process.split(',')]

    for length in code_lengths_to_run:
        print(f"\n\n>>>>>>>>>> STARTING WORKFLOW FOR CODE LENGTH: {length} <<<<<<<<<<")
        
        # Create a new opt object for this specific run, copying master settings
        # and setting the current code_length. This is crucial because models
        # in the original script use the global 'opt.code_length'.
        current_run_options = argparse.Namespace(**vars(master_opt_namespace))
        current_run_options.code_length = length # Set the code_length for this specific run
        
        # The train_and_generate_for_single_length function now expects 'opt_config'
        # which will be this current_run_options.
        # The models Generator() and Discriminator() will be instantiated inside that function,
        # and their __init__ will refer to current_run_options.code_length etc.
        
        train_and_generate_for_single_length(current_run_options)
            
    print("\n===== ALL SPECIFIED CODE LENGTHS PROCESSED. SCRIPT FINISHED. =====")