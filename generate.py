import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

FONT_PATH = "arial.ttf"  # 或改成絕對路徑
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
FONT_SIZE = 40
NUM_CLASSES = [4, 5, 6, 7]

def random_code(length):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

def create_image(code, font_path='arial.ttf'):
    width, height = IMAGE_WIDTH, IMAGE_HEIGHT
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    x_offset = 10  # 起始貼字位置

    for ch in code:
        font_size = random.randint(30, 45)
        font = ImageFont.truetype(font_path, font_size)

        char_img = Image.new('RGBA', (60, 60), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((5, 5), ch, font=font, fill=(0, 0, 0))

        angle = random.randint(-30, 30)
        char_img = char_img.rotate(angle, resample=Image.BICUBIC, expand=1)

        def _shear(img):
            m = -0.3 + random.random() * 0.6
            return img.transform(
                img.size,
                Image.AFFINE,
                (1, m, 0, 0, 1, 0),
                resample=Image.BICUBIC,
                fillcolor=(255, 255, 255, 0)
            )
        char_img = _shear(char_img)

        y_offset = random.randint(5, 15)
        image.paste(char_img, (x_offset, y_offset), char_img)

        # 間距設為字寬 - 15，字會稍微重疊更緊湊
        x_offset += char_img.size[0] - 15

    # 加雜訊線
    for _ in range(15):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line([(x1, y1), (x2, y2)], fill=(random.randint(0, 200),)*3, width=random.randint(1, 3))

    # 加雜訊點
    for _ in range(1000):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        image.putpixel((x, y), (random.randint(0, 200),)*3)

    image = image.filter(ImageFilter.GaussianBlur(1.2))

    return image



def generate_dataset(total_count):
    count_per_class = total_count // len(NUM_CLASSES)
    os.makedirs("data", exist_ok=True)

    for length in NUM_CLASSES:
        class_dir = os.path.join("data", str(length))
        os.makedirs(class_dir, exist_ok=True)
        for i in range(count_per_class):
            code = random_code(length)
            img = create_image(code, "arial.ttf")
            img.save(os.path.join(class_dir, f"{length}_{i}_{code}.png"))

if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    generate_dataset(num)
