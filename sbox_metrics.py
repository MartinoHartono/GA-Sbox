"""
Metrik kriptografis untuk S-Box 8x8 (256 entri, nilai 0-255).

Yang diimplementasikan:
  - Balance & Bijective  (syarat awal)
  - Nonlinearity (NL)
  - SAC  (Strict Avalanche Criterion)
  - BIC-NL  (Bit Independence Criterion - Nonlinearity)
  - BIC-SAC (Bit Independence Criterion - SAC)
  - LAP  (Linear Approximation Probability)
  - DAP  (Differential Approximation Probability)  + DU (differential uniformity)
  - Tambahan: fixed point & opposite fixed point

Semua dihitung dengan numpy (Walsh-Hadamard lewat perkalian matriks).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

_IDX = np.arange(256)
PARITY = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.int8)

# Matriks Hadamard H[a, x] = (-1)^(a . x)  (simetris)
HAD = (1 - 2 * PARITY[np.bitwise_and.outer(_IDX, _IDX)]).astype(np.float32)

# Mask output untuk pasangan bit (j, k) pada BIC
BIT_PAIRS = list(combinations(range(8), 2))
PAIR_MASKS = np.array([(1 << j) | (1 << k) for j, k in BIT_PAIRS])


# --------------------------------------------------------------------------
# Utilitas dasar
# --------------------------------------------------------------------------
def as_sbox(s) -> np.ndarray:
    """Terima list/array 256 entri atau matriks 16x16 -> array int64 panjang 256."""
    arr = np.asarray(s, dtype=np.int64).reshape(-1)
    if arr.size != 256:
        raise ValueError("S-Box harus berisi 256 nilai (16x16).")
    return arr


def is_bijective(s) -> bool:
    s = as_sbox(s)
    return bool(s.min() >= 0 and s.max() <= 255 and len(np.unique(s)) == 256)


def balance_counts(s) -> list[int]:
    """Jumlah bit '1' pada tiap bit output (ideal: 128 untuk masing-masing)."""
    s = as_sbox(s)
    return [int(((s >> j) & 1).sum()) for j in range(8)]


def is_balanced(s) -> bool:
    return all(c == 128 for c in balance_counts(s))


# --------------------------------------------------------------------------
# Spektrum Walsh -> NL, LAP
# --------------------------------------------------------------------------
def walsh_all(s) -> np.ndarray:
    """W[b, a] = sum_x (-1)^(b.S(x) xor a.x), untuk semua mask output b dan input a.
    Baris b=0 tidak bermakna (fungsi konstan) dan diabaikan oleh pemanggil."""
    s = as_sbox(s)
    F = (1 - 2 * PARITY[np.bitwise_and.outer(_IDX, s)]).astype(np.float32)  # (b, x)
    return F @ HAD  # (b, a)  -- eksak karena nilai bulat kecil


def component_nl(W: np.ndarray) -> np.ndarray:
    """NL tiap fungsi komponen b (indeks 0 tidak dipakai)."""
    return 128 - np.abs(W).max(axis=1) / 2.0


def nonlinearity(s, W: np.ndarray | None = None) -> int:
    """NL S-Box = min NL semua kombinasi linear bit output (b != 0)."""
    W = walsh_all(s) if W is None else W
    return int(round(128 - np.abs(W[1:]).max() / 2.0))


def lap(s, W: np.ndarray | None = None) -> float:
    """LAP = max_{a,b != 0} | #{x: a.x = b.S(x)} - 128 | / 256  =  max|W| / 512."""
    W = walsh_all(s) if W is None else W
    return float(np.abs(W[1:, 1:]).max() / 512.0)


# --------------------------------------------------------------------------
# SAC dan BIC
# --------------------------------------------------------------------------
def sac_matrix(s) -> np.ndarray:
    """M[i, j] = P( bit output j berubah | bit input i dibalik ). Ideal 0.5."""
    s = as_sbox(s)
    flips = 1 << np.arange(8)                                   # (8,)
    D = s[None, :] ^ s[_IDX[None, :] ^ flips[:, None]]          # (i, x)
    bits = (D[:, :, None] >> np.arange(8)) & 1                  # (i, x, j)
    return bits.mean(axis=1)


def bic_sac(s) -> dict:
    """SAC dari fungsi f_j XOR f_k untuk semua pasangan bit output (j<k)."""
    s = as_sbox(s)
    vals = np.zeros((8, len(BIT_PAIRS)))
    for i in range(8):
        D = s ^ s[_IDX ^ (1 << i)]
        for p, (j, k) in enumerate(BIT_PAIRS):
            vals[i, p] = (((D >> j) ^ (D >> k)) & 1).mean()
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
            "matrix": vals}


def bic_nl(W: np.ndarray) -> dict:
    cnl = component_nl(W)
    v = cnl[PAIR_MASKS]
    return {"min": int(round(v.min())), "mean": float(v.mean()), "max": int(round(v.max()))}


# --------------------------------------------------------------------------
# Diferensial
# --------------------------------------------------------------------------
def ddt(s) -> np.ndarray:
    """Difference Distribution Table (256x256). Baris dx=0 bernilai trivial."""
    s = as_sbox(s)
    dx = np.arange(256)[:, None]
    dy = s[None, :] ^ s[_IDX[None, :] ^ dx]            # (dx, x)
    flat = (dx * 256 + dy).ravel()
    return np.bincount(flat, minlength=65536).reshape(256, 256)


def dap(s, T: np.ndarray | None = None) -> float:
    """DAP = max_{dx != 0, dy} DDT[dx, dy] / 256."""
    T = ddt(s) if T is None else T
    return float(T[1:].max() / 256.0)


def differential_uniformity(s, T: np.ndarray | None = None) -> int:
    T = ddt(s) if T is None else T
    return int(T[1:].max())


# --------------------------------------------------------------------------
# Tambahan
# --------------------------------------------------------------------------
def fixed_points(s) -> tuple[int, int]:
    s = as_sbox(s)
    return int((s == _IDX).sum()), int((s == (_IDX ^ 255)).sum())


# --------------------------------------------------------------------------
# Evaluasi lengkap
# --------------------------------------------------------------------------
def evaluate(s) -> dict:
    """Hitung semua metrik; hasil berupa dict yang siap ditampilkan."""
    s = as_sbox(s)
    W = walsh_all(s)
    T = ddt(s)
    sac = sac_matrix(s)
    bsac = bic_sac(s)
    bnl = bic_nl(W)
    fp, ofp = fixed_points(s)
    return {
        "bijective": is_bijective(s),
        "balanced": is_balanced(s),
        "balance_counts": balance_counts(s),
        "NL": nonlinearity(s, W),
        "SAC_mean": float(sac.mean()),
        "SAC_min": float(sac.min()),
        "SAC_max": float(sac.max()),
        "SAC_matrix": sac,
        "BIC_NL_min": bnl["min"],
        "BIC_NL_mean": bnl["mean"],
        "BIC_SAC_mean": bsac["mean"],
        "BIC_SAC_min": bsac["min"],
        "BIC_SAC_max": bsac["max"],
        "LAP": lap(s, W),
        "DAP": dap(s, T),
        "DU": differential_uniformity(s, T),
        "fixed_points": fp,
        "opposite_fixed_points": ofp,
    }


def summary_table(m: dict) -> list[tuple[str, str, str]]:
    """Baris tabel ringkas: (kriteria, nilai, target)."""
    return [
        ("Bijective", "Ya" if m["bijective"] else "Tidak", "Ya"),
        ("Balance (tiap bit 128:128)", "Ya" if m["balanced"] else "Tidak", "Ya"),
        ("Nonlinearity (NL)", f"{m['NL']}", ">= 112"),
        ("SAC (rata-rata)", f"{m['SAC_mean']:.5f}", "≈ 0.5"),
        ("BIC-NL (min)", f"{m['BIC_NL_min']}", "112"),
        ("BIC-SAC (rata-rata)", f"{m['BIC_SAC_mean']:.5f}", "≈ 0.5"),
        ("LAP", f"{m['LAP']:.5f}", "→ 0"),
        ("DAP", f"{m['DAP']:.5f}", "→ 0"),
        ("Differential uniformity (DU)", f"{m['DU']}", "4 (AES)"),
        ("Fixed point / opposite", f"{m['fixed_points']} / {m['opposite_fixed_points']}", "0 / 0"),
    ]


# --------------------------------------------------------------------------
# S-Box AES standar (pembanding & uji kebenaran metrik)
# --------------------------------------------------------------------------
def _gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11B
        b >>= 1
    return r


def aes_sbox() -> np.ndarray:
    inv = [0] * 256
    for x in range(1, 256):
        for y in range(1, 256):
            if _gmul(x, y) == 1:
                inv[x] = y
                break
    out = []
    for x in range(256):
        b = inv[x]
        r = b
        for sh in range(1, 5):
            r ^= ((b << sh) | (b >> (8 - sh))) & 0xFF
        out.append(r ^ 0x63)
    return np.array(out, dtype=np.int64)


def to_matrix(s) -> np.ndarray:
    return as_sbox(s).reshape(16, 16)
