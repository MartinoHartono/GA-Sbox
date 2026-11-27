"""Membuat 3 gambar uji sintetis (512x512 RGB) di folder dataset/ bila folder itu kosong.
Gambar asli (cameraman, lena_color_512, livingroom, mandril_color, pirate, woman_blonde,
woman_darkhair) cukup disalin ke folder dataset/ -- aplikasi dan pipeline akan memakainya."""
from pathlib import Path

import numpy as np
from PIL import Image

from image_crypto import list_dataset


def synth(seed: int, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    base = np.stack([np.sin(3 * np.pi * x + seed) * 0.5 + 0.5,
                     np.cos(2 * np.pi * y + seed) * 0.5 + 0.5,
                     (x + y) / 2], axis=-1)
    img = base.copy()
    for _ in range(14):                                   # bentuk persegi / lingkaran berwarna
        cx, cy, r = rng.random(), rng.random(), rng.uniform(0.04, 0.2)
        col = rng.random(3)
        mask = ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2) if rng.random() < 0.5 \
            else ((abs(x - cx) < r) & (abs(y - cy) < r))
        img[mask] = col
    img += rng.normal(0, 0.02, img.shape)                  # sedikit tekstur
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main(folder="dataset"):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    if list_dataset(folder):
        print("dataset/ sudah berisi gambar, tidak membuat gambar sintetis.")
        return
    for i, name in enumerate(["sintetis_1", "sintetis_2", "sintetis_3"], start=1):
        Image.fromarray(synth(i)).save(folder / f"{name}.png")
    print("3 gambar sintetis dibuat di", folder)


if __name__ == "__main__":
    main()
