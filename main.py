import json
import math
import random
from PIL import Image, ImageDraw
import numpy as np
from PIL.ImageFilter import GaussianBlur

STROKES = {
    "0": [
        [[(14,4), (20,8), (22,14), (20,20), (14,24), (8,20), (6,14), (8,8), (14,4)]]
    ],
    "1": [
        [[(14,4), (13,14), (15,24)]]
    ],
    "2": [
        [[(7,7), (14,4), (20,8), (18,14), (8,22), (22,24)]]
    ],
    "3": [
        [[(7,7), (15,4), (20,9), (15,13), (20,18), (15,24), (7,21)]]
    ],
    "4": [
        [
            [(16,4), (6,18), (20,18)],   # diagonal down then across
            [(16,4), (16,24)]             # vertical stem
        ]
    ],
    "5": [
        [
            [(20,5), (7,5)],                      # top bar
            [(7,5), (7,13), (15,13)],              # down then across
            [(15,13), (20,18), (15,23), (7,21)]    # curve into bottom bump
        ]
    ],
    "6": [
        [[(19,5), (10,8), (7,16), (10,23), (17,23), (20,18), (16,14), (9,15)]]
    ],
    "7": [
        [
            [(5,5), (23,5)],
            [(22,6), (15,14), (10,24)]
        ]
    ],
    "8": [
        [
            [(14,4), (18,5), (19,8), (17,10), (14,11), (11,10), (9,8), (10,5), (14,4)],     # top loop
            [(14,11), (19,14), (20,18), (18,22), (14,24), (10,22), (8,18), (9,14), (14,11)] # bottom loop
        ]
    ],
    "9": [
        [[(18,9), (13,5), (8,9), (13,13), (18,9), (19,14), (16,19), (12,24)]],  # variant 1: curved hook tail
        [[(18,9), (13,5), (8,9), (13,13), (18,9), (17,16), (16,24)]]            # variant 2: straight tail
    ]
}

def send(msg):
    print(json.dumps(msg), flush=True)

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

    variant = random.choice(STROKES[digit])

    for stroke in variant:
        jittered_points = [jitter_point(point=p) for p in stroke]
        thickness = random.randint(1, 2)
        draw.line(jittered_points, fill='white', width=thickness)

    canvas = apply_blur(canvas, random.uniform(c['blur_range'][0], c['blur_range'][1]))
    canvas = apply_gaussian_noise(canvas, c['noise_intensity'])
    return canvas

def generate_dataset(c):
    all_images = []
    total = sum(c['distribution'].values())
    completed = 0

    for digit, count in c['distribution'].items():
        for i in range(count):
            img = generate_digit_image(digit, c)
            all_images.append((img, digit, i))
            completed += 1
            send({"status": "progress", "completed": completed, "total": total})

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
        "distribution": { "0": 10, "1": 10, "2": 10, "3": 10, "4": 10, "5": 10, "6": 10, "7": 10, "8": 10, "9": 10 },
        "noise_intensity": 10,
        "blur_range": [0.3, 0.8]
    }

    all_images = generate_dataset(config)
    train, val = split_dataset(all_images)
    save_dataset(train, val, "output")
    send({"status": "done", "train_count": len(train), "val_count": len(val), "output_dir": "output"})




if __name__ == "__main__":
    main()