"""
Enkripsi/dekripsi gambar dengan AES ber-S-Box kustom + metrik kualitas enkripsi:
Entropy, Histogram (+ chi-square), NPCR, UACI, korelasi piksel bertetangga.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

from aes_custom import AES

IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


# ------------------------------------------------------------------ I/O gambar
def load_image(src) -> np.ndarray:
    """Path / bytes / file-like -> array uint8 (H,W) untuk grayscale atau (H,W,3) untuk warna."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    im = Image.open(src)
    im = im.convert("L") if im.mode in ("1", "L") else im.convert("RGB")
    return np.array(im, dtype=np.uint8)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def list_dataset(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


# --------------------------------------------------------- enkripsi / dekripsi
def encrypt_image(img: np.ndarray, aes: AES, mode: str = "CBC", iv: bytes | None = None):
    """Return (cipher_img, cipher_bytes). cipher_img berukuran sama dengan img (untuk ditampilkan);
    cipher_bytes (lengkap, mungkin berpadding nol) dipakai untuk dekripsi."""
    raw = img.tobytes()
    cb = aes.encrypt(raw, mode, iv)
    cimg = np.frombuffer(cb[:len(raw)], dtype=np.uint8).reshape(img.shape).copy()
    return cimg, cb


def decrypt_image(cipher_bytes: bytes, shape: tuple, aes: AES, mode: str = "CBC",
                  iv: bytes | None = None) -> np.ndarray:
    n = int(np.prod(shape))
    plain = aes.decrypt(cipher_bytes, n, mode, iv)
    return np.frombuffer(plain, dtype=np.uint8).reshape(shape).copy()


# ------------------------------------------------------------------- metrik
def entropy(img: np.ndarray) -> float:
    """Entropi Shannon (bit/piksel) atas seluruh byte. Ideal ≈ 8."""
    cnt = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def entropy_per_channel(img: np.ndarray) -> list[float]:
    if img.ndim == 2:
        return [entropy(img)]
    return [entropy(img[:, :, c]) for c in range(img.shape[2])]


def histograms(img: np.ndarray) -> list[np.ndarray]:
    if img.ndim == 2:
        return [np.bincount(img.reshape(-1), minlength=256)]
    return [np.bincount(img[:, :, c].reshape(-1), minlength=256) for c in range(img.shape[2])]


def chi_square(img: np.ndarray) -> tuple[float, float]:
    """Uji keseragaman histogram (semua byte). Seragam jika p-value > 0.05 (chi2 < 293.25, df=255)."""
    cnt = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    exp = cnt.sum() / 256.0
    chi2 = float(((cnt - exp) ** 2 / exp).sum())
    return chi2, float(stats.chi2.sf(chi2, 255))


def npcr_uaci(c1: np.ndarray, c2: np.ndarray) -> tuple[float, float]:
    """NPCR & UACI (%) antara dua citra cipher berukuran sama."""
    a = c1.astype(np.int16).reshape(-1)
    b = c2.astype(np.int16).reshape(-1)
    npcr = float((a != b).mean() * 100.0)
    uaci = float((np.abs(a - b) / 255.0).mean() * 100.0)
    return npcr, uaci


def adjacent_correlation(img: np.ndarray, direction: str = "h", n: int = 5000, seed: int = 0) -> float:
    """Koefisien korelasi pasangan piksel bertetangga (channel pertama / gray). Plain≈1, cipher≈0."""
    g = img if img.ndim == 2 else img[:, :, 0]
    h, w = g.shape
    rng = np.random.default_rng(seed)
    if direction == "h":
        ys, xs = rng.integers(0, h, n), rng.integers(0, w - 1, n)
        a, b = g[ys, xs], g[ys, xs + 1]
    elif direction == "v":
        ys, xs = rng.integers(0, h - 1, n), rng.integers(0, w, n)
        a, b = g[ys, xs], g[ys + 1, xs]
    else:
        ys, xs = rng.integers(0, h - 1, n), rng.integers(0, w - 1, n)
        a, b = g[ys, xs], g[ys + 1, xs + 1]
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def one_pixel_change_test(img: np.ndarray, aes: AES, mode: str = "CBC", iv: bytes | None = None,
                          n_random: int = 3, seed: int = 0) -> dict:
    """Ubah 1 piksel (balik 1 bit terendah satu channel) pada plain, enkripsi ulang, bandingkan cipher.
    - first : piksel pertama (efek maksimum pada mode berantai seperti CBC)
    - random: rata-rata dari n_random posisi acak
    Catatan: pada ECB perubahan 1 piksel hanya memengaruhi 1 blok (16 byte) -> NPCR kecil; itu sifat
    mode, bukan kelemahan S-Box. Untuk uji difusi gunakan CBC."""
    base, _ = encrypt_image(img, aes, mode, iv)
    rng = np.random.default_rng(seed)
    flat_n = img.size
    positions = [0] + [int(rng.integers(0, flat_n)) for _ in range(n_random)]
    res = []
    for pos in positions:
        mod = img.copy().reshape(-1)
        mod[pos] ^= 1
        c2, _ = encrypt_image(mod.reshape(img.shape), aes, mode, iv)
        res.append(npcr_uaci(base, c2))
    first = res[0]
    rnd = np.mean(res[1:], axis=0) if len(res) > 1 else first
    return {"NPCR_first": first[0], "UACI_first": first[1],
            "NPCR_random": float(rnd[0]), "UACI_random": float(rnd[1])}


def analyze_image(img: np.ndarray, aes: AES, mode: str = "CBC", iv: bytes | None = None,
                  n_random: int = 3, seed: int = 0) -> dict:
    """Satu paket: enkripsi, dekripsi, dan seluruh metrik."""
    cimg, cb = encrypt_image(img, aes, mode, iv)
    dimg = decrypt_image(cb, img.shape, aes, mode, iv)
    chi2, p = chi_square(cimg)
    out = {
        "shape": img.shape,
        "entropy_plain": entropy(img),
        "entropy_cipher": entropy(cimg),
        "entropy_cipher_channels": entropy_per_channel(cimg),
        "chi2_cipher": chi2,
        "chi2_pvalue": p,
        "corr_h_plain": adjacent_correlation(img, "h"),
        "corr_h_cipher": adjacent_correlation(cimg, "h"),
        "decrypt_identical": bool(np.array_equal(dimg, img)),
        "decrypt_max_abs_diff": int(np.abs(dimg.astype(int) - img.astype(int)).max()),
    }
    out.update(one_pixel_change_test(img, aes, mode, iv, n_random, seed))
    return out, cimg, dimg
