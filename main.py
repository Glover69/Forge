import math
import random
from PIL import Image, ImageDraw
import numpy as np
from PIL.ImageFilter import GaussianBlur

STROKES = {
    "1": [[(14,4), (13,14), (15,24)]],
    "7": [[(5,5), (23,5)], [(22,6), (15,14), (10,24)]]
}

def jitter_point(point, max_offset=1.5):
    dx = random.uniform(-max_offset, max_offset)
    dy = random.uniform(-max_offset, max_offset)
    return point[0] + dx, point[1] + dy

def apply_blur(image, radius):
    return image.filter(GaussianBlur(radius))

def apply_gaussian_noise(image, intensity):
    pixels = np.asarray(image)
    noise = np.random.default_rng().normal(0, intensity, pixels.shape)
    noisy = pixels + noise
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)  # convert back to valid pixel type
    return Image.fromarray(noisy)


def generate_digit_image(digit, c):
    canvas = Image.new("RGB", (28, 28), "black")
    draw = ImageDraw.Draw(canvas)

    for stroke in STROKES[digit]:
        jittered_points = [jitter_point(point=p) for p in stroke]
        thickness = random.randint(1, 2)
        draw.line(jittered_points, fill='white', width=thickness)

    canvas = apply_blur(canvas, random.uniform(c['blur_range'][0], c['blur_range'][1]))
    canvas = apply_gaussian_noise(canvas, c['noise_intensity'])
    return canvas

def generate_dataset(c):
    all_images = []

    for digit, count in c['distribution'].items():
        for i in range(count):
            img = generate_digit_image(digit, c)
            all_images.append((img, digit, i))

    return all_images

def split_dataset(images, train_ratio= 0.8):
    train = []
    val = []

    groups = {}

    for item in images:
        img, digit, i = item
        if digit not in groups:
            groups[digit] = []

        groups[digit].append(item)


    for digit, images in groups.items():
        random.shuffle(images)
        split_point = math.floor(len(images) * train_ratio)
        train += images[:split_point]
        val += images[split_point:]

    return train, val


def save_dataset(train, val, output_dir):
    for img, digit, i in train:
        img.save(f"{output_dir}/train/{digit}_{i}.jpg")
    for img, digit, i in val:
        img.save(f"{output_dir}/val/{digit}_{i}.jpg")


def main():
    config = {
        "distribution": {"1": 20, "7": 20},
        "noise_intensity": 10,
        "blur_range": [0.3, 0.8]
    }

    all_images = generate_dataset(config)
    train, val = split_dataset(all_images)
    save_dataset(train, val, "output")


if __name__ == "__main__":
    main()