"""Uji regresi cepat:  python test_all.py   (±20 detik)"""
import os
from pathlib import Path

import numpy as np

import image_crypto as IC
import sbox_metrics as M
from aes_custom import AES
from sbox_ga import (AffineGAConfig, GAConfig, _affine_mutate, mat_invertible, order_crossover,
                     random_invertible, run_affine_ga, run_ga, build_affine_sbox)
from sbox_io import load_sbox

ok_count = 0


def check(name, cond):
    global ok_count
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    ok_count += 1


# --- 1. metrik: S-Box AES harus menghasilkan nilai literatur
m = M.evaluate(M.aes_sbox())
check("AES S-Box: NL=112, DU=4, BIC-NL=112", (m["NL"], m["DU"], m["BIC_NL_min"]) == (112, 4, 112))
check("AES S-Box: LAP=0.0625, DAP=0.015625", abs(m["LAP"] - 0.0625) < 1e-9 and abs(m["DAP"] - 4 / 256) < 1e-9)
check("AES S-Box: SAC≈0.50488, BIC-SAC≈0.50460", abs(m["SAC_mean"] - 0.50488) < 1e-4 and abs(m["BIC_SAC_mean"] - 0.50460) < 1e-4)
check("Identitas (S(x)=x) tidak balance-nonlinear: NL=0", M.nonlinearity(np.arange(256)) == 0)

# --- 2. AES: test vector FIPS-197 + pembanding library cryptography
pt = bytes.fromhex("00112233445566778899aabbccddeeff")
k128 = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
check("AES-128 FIPS-197", AES(k128).encrypt(pt, "ECB").hex() == "69c4e0d86a7b0430d8cdb78070b4c55a")
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    rng = np.random.default_rng(0)
    data, iv = rng.integers(0, 256, 16 * 200, dtype=np.uint8).tobytes(), os.urandom(16)
    for kl in (16, 24, 32):
        k = os.urandom(kl)
        e = Cipher(algorithms.AES(k), modes.CBC(iv)).encryptor()
        check(f"AES-{kl * 8} CBC == library cryptography", AES(k).encrypt(data, "CBC", iv) == e.update(data) + e.finalize())
except ImportError:
    print("SKIP  perbandingan dengan library cryptography (tidak terpasang)")

# --- 3. operator GA
rng = np.random.default_rng(1)
ok = all(np.array_equal(np.sort(order_crossover(rng.permutation(256), rng.permutation(256), rng)), np.arange(256)) for _ in range(200))
check("Order crossover selalu menghasilkan permutasi valid", ok)
A = random_invertible(rng); B = random_invertible(rng)
ind = (A, B, 5)
for _ in range(300):
    ind = _affine_mutate(ind, rng)
check("Mutasi afin menjaga invertibilitas matriks", mat_invertible(ind[0]) and mat_invertible(ind[1]))
s = build_affine_sbox(*ind)
check("S-Box afin: bijective & balance", M.is_bijective(s) and M.is_balanced(s))
check("S-Box afin: NL=112, DU=4 (invarian afin)", M.nonlinearity(s) == 112 and M.differential_uniformity(s) == 4)

r = run_ga(GAConfig(pop_size=10, generations=3, ls_steps=5, seed=0))
check("GA permutasi (run singkat): bijective & balance", M.is_bijective(r.best) and M.is_balanced(r.best))
r2 = run_affine_ga(AffineGAConfig(pop_size=10, generations=3, seed=0))
check("GA afin (run singkat): bijective & balance", M.is_bijective(r2.best) and M.is_balanced(r2.best))

# --- 4. hasil tersimpan + enkripsi/dekripsi gambar
img = np.random.default_rng(2).integers(0, 256, (37, 53, 3), dtype=np.uint8)       # ukuran bukan kelipatan 16
key, iv = bytes(range(16)), bytes(range(16, 32))
cands = {"AES": M.aes_sbox()}
for f in ("sbox_ga_permutasi.json", "sbox_ga_afin.json"):
    p = Path("results") / f
    if p.exists():
        d = load_sbox(p)
        check(f"{f}: bijective & balance", M.is_bijective(d["sbox"]) and M.is_balanced(d["sbox"]))
        cands[f] = d["sbox"]
for nm, sb in cands.items():
    for mode in ("ECB", "CBC"):
        a = AES(key, sb)
        c, cb = IC.encrypt_image(img, a, mode, iv)
        dec = IC.decrypt_image(cb, img.shape, a, mode, iv)
        check(f"enkripsi->dekripsi identik [{nm}, {mode}]", np.array_equal(dec, img))
a = AES(key, cands["AES"])
bad = AES(bytes(range(1, 17)), cands["AES"])
_, cb = IC.encrypt_image(img, a, "CBC", iv)
check("kunci salah -> dekripsi gagal", not np.array_equal(IC.decrypt_image(cb, img.shape, bad, "CBC", iv), img))
print(f"\nSemua {ok_count} pemeriksaan lolos.")
