"""
Pipeline lengkap (tanpa Streamlit):
  1. GA mode A  : S-Box dari permutasi 0..255
  2. GA mode B  : S-Box dari parameter afin atas invers GF(2^8)
  3. Uji S-Box  : NL, SAC, BIC-NL, BIC-SAC, LAP, DAP (+DU, fixed point) vs AES standar
  4. Enkripsi/dekripsi semua gambar di dataset/ (AES standar vs S-Box GA), uji Entropy/Histogram/NPCR/UACI
Hasil -> folder results/

Contoh:
    python run_pipeline.py                         # default
    python run_pipeline.py --gens-a 600 --pop-a 80
    python run_pipeline.py --skip-ga               # pakai S-Box yang sudah tersimpan di results/
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import image_crypto as IC
import sbox_metrics as M
from aes_custom import AES, key_to_bytes
from make_sample_images import main as make_samples
from sbox_ga import AffineGAConfig, GAConfig, run_affine_ga, run_ga
from sbox_io import load_sbox, save_sbox

RES = Path("results")
DEFAULT_KEY = "2b7e151628aed2a6abf7158809cf4f3c"
DEFAULT_IV = bytes(range(16))


def plot_history(histories: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, h in histories.items():
        g = [r["gen"] for r in h]
        axes[0].plot(g, [r["cost"] for r in h], label=name)
        axes[1].plot(g, [r["NL"] for r in h], label=name)
    axes[0].set_title("Konvergensi biaya (lebih kecil lebih baik)")
    axes[1].set_title("Nonlinearity individu terbaik")
    for a in axes:
        a.set_xlabel("generasi")
        a.grid(alpha=.3)
        a.legend()
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_hist(img, cimg, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].imshow(img, cmap="gray" if img.ndim == 2 else None); axes[0, 0].set_title("Plain")
    axes[0, 1].imshow(cimg, cmap="gray" if cimg.ndim == 2 else None); axes[0, 1].set_title("Cipher")
    cols = ["r", "g", "b"] if img.ndim == 3 else ["k"]
    for k, (h, c) in enumerate(zip(IC.histograms(img), cols)):
        axes[1, 0].plot(h, color=c, lw=.8)
    for k, (h, c) in enumerate(zip(IC.histograms(cimg), cols)):
        axes[1, 1].plot(h, color=c, lw=.8)
    axes[1, 0].set_title("Histogram plain"); axes[1, 1].set_title("Histogram cipher")
    axes[1, 1].set_ylim(0, max(1, max(h.max() for h in IC.histograms(cimg))) * 2)
    for a in axes[0]:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop-a", type=int, default=60)
    ap.add_argument("--gens-a", type=int, default=400)
    ap.add_argument("--pop-b", type=int, default=100)
    ap.add_argument("--gens-b", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ga", action="store_true")
    ap.add_argument("--mode", default="CBC", choices=["CBC", "ECB"])
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args()

    RES.mkdir(exist_ok=True)
    pa, pb = RES / "sbox_ga_permutasi.json", RES / "sbox_ga_afin.json"

    # ---------------------------------------------------------------- 1-2. GA
    if not args.skip_ga:
        t = time.time()
        print(f"[A] GA permutasi: pop={args.pop_a} gen={args.gens_a}", flush=True)
        cfg = GAConfig(pop_size=args.pop_a, generations=args.gens_a, seed=args.seed)
        ra = run_ga(cfg, lambda g, r: print(f"  gen {g:4d} cost {r['cost']:.4f} NL {r['NL']} DU {r['DU']}", flush=True)
                    if g % 50 == 0 else None)
        save_sbox(pa, "S-Box GA (permutasi)", "GA permutasi 0..255 + memetic local search", ra.best,
                  M.evaluate(ra.best), ra.config, ra.seconds, ra.history)
        print(f"[A] selesai {time.time() - t:.0f}s", flush=True)

        print(f"[B] GA afin: pop={args.pop_b} gen={args.gens_b}", flush=True)
        cfgb = AffineGAConfig(pop_size=args.pop_b, generations=args.gens_b, seed=args.seed)
        rb = run_affine_ga(cfgb, lambda g, r: print(f"  gen {g:4d} cost {r['cost']:.4f} NL {r['NL']} DU {r['DU']}", flush=True)
                           if g % 50 == 0 else None)
        save_sbox(pb, "S-Box GA (afin atas invers GF(2^8))", "GA parameter afin (A,B,c) pada invers GF(2^8)",
                  rb.best, M.evaluate(rb.best), rb.config, rb.seconds, rb.history)

    A, B = load_sbox(pa), load_sbox(pb)
    plot_history({"Mode A (permutasi)": A["history"], "Mode B (afin)": B["history"]}, RES / "konvergensi_ga.png")

    # ---------------------------------------------------------------- 3. Uji S-Box
    candidates = {"AES standar": M.aes_sbox(), "GA mode A (permutasi)": A["sbox"], "GA mode B (afin)": B["sbox"]}
    mets = {k: M.evaluate(v) for k, v in candidates.items()}
    rows = [("Kriteria", "Target") + tuple(candidates)]
    keys = [("Bijective", "bijective", "Ya"), ("Balance", "balanced", "Ya"), ("NL", "NL", ">=112"),
            ("SAC mean", "SAC_mean", "0.5"), ("BIC-NL min", "BIC_NL_min", "112"), ("BIC-SAC mean", "BIC_SAC_mean", "0.5"),
            ("LAP", "LAP", "->0"), ("DAP", "DAP", "->0"), ("DU", "DU", "4"),
            ("Fixed point", "fixed_points", "0")]
    for label, k, tg in keys:
        rows.append((label, tg) + tuple((f"{mets[n][k]:.5f}" if isinstance(mets[n][k], float) else str(mets[n][k]))
                                        for n in candidates))
    with open(RES / "metrik_sbox.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("\n=== Metrik S-Box ===")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:>7s} | " + " | ".join(f"{v:>12s}" for v in r[2:]))

    # ---------------------------------------------------------------- 4. Enkripsi gambar
    make_samples(args.dataset)
    files = IC.list_dataset(args.dataset)
    key = key_to_bytes(args.key, 128)
    out_rows = [("gambar", "S-Box", "mode", "entropy_cipher", "chi2", "chi2_p", "NPCR_first_%", "UACI_first_%",
                 "NPCR_rand_%", "UACI_rand_%", "corr_h_plain", "corr_h_cipher", "dekripsi_identik")]
    (RES / "gambar").mkdir(exist_ok=True)
    print(f"\n=== Enkripsi gambar ({args.mode}), {len(files)} gambar ===")
    for fp in files:
        img = IC.load_image(fp)
        for nm, sb in candidates.items():
            r, cimg, dimg = IC.analyze_image(img, AES(key, sb), args.mode, DEFAULT_IV, n_random=3)
            out_rows.append((fp.stem, nm, args.mode, f"{r['entropy_cipher']:.5f}", f"{r['chi2_cipher']:.1f}",
                             f"{r['chi2_pvalue']:.3f}", f"{r['NPCR_first']:.4f}", f"{r['UACI_first']:.4f}",
                             f"{r['NPCR_random']:.4f}", f"{r['UACI_random']:.4f}", f"{r['corr_h_plain']:.4f}",
                             f"{r['corr_h_cipher']:.4f}", r["decrypt_identical"]))
            print(f"  {fp.stem:18s} {nm:22s} H={r['entropy_cipher']:.5f} NPCR={r['NPCR_first']:.3f}% "
                  f"UACI={r['UACI_first']:.3f}% dekripsi_identik={r['decrypt_identical']}", flush=True)
            if nm == "GA mode B (afin)":
                Image.fromarray(cimg).save(RES / "gambar" / f"{fp.stem}_cipher.png")
                Image.fromarray(dimg).save(RES / "gambar" / f"{fp.stem}_dekripsi.png")
                plot_hist(img, cimg, f"{fp.stem} - {nm}", RES / "gambar" / f"{fp.stem}_histogram.png")
    with open(RES / "metrik_gambar.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)
    print("\nSelesai. Lihat folder results/")


if __name__ == "__main__":
    main()
