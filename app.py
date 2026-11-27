"""
Aplikasi Streamlit: Konstruksi S-Box dengan Algoritma Genetika + AES modifikasi untuk enkripsi gambar.
Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import image_crypto as IC
import sbox_metrics as M
from aes_custom import AES, key_to_bytes
from sbox_ga import AffineGAConfig, GAConfig, run_affine_ga, run_ga
from sbox_io import load_sbox, parse_sbox_text, sbox_matrix_hex, sbox_to_csv, save_sbox

RES = Path("results")
DATASET = Path("dataset")
DEFAULT_IV = bytes(range(16))

st.set_page_config(page_title="S-Box GA + AES Image Encryption", layout="wide")
st.title("S-Box dengan Algoritma Genetika → AES Modifikasi → Enkripsi Gambar")
st.caption("Kriptografi TI – Proyek S-Box. Alur: (1) bangun S-Box dengan GA, (2) uji kualitas S-Box, "
           "(3) ganti S-Box AES, enkripsi & dekripsi gambar, uji hasil.")

# ---------------------------------------------------------------- state awal
if "sbox" not in st.session_state:
    saved = RES / "sbox_ga_afin.json"
    if saved.exists():
        d = load_sbox(saved)
        st.session_state.sbox, st.session_state.sbox_name = d["sbox"], d["name"]
        st.session_state.history = d.get("history", [])
    else:
        st.session_state.sbox, st.session_state.sbox_name = M.aes_sbox(), "AES standar"
        st.session_state.history = []


def show_sbox_table(sbox):
    hexm = sbox_matrix_hex(sbox)
    lab = [f"{i:X}" for i in range(16)]
    st.dataframe(pd.DataFrame(hexm, index=lab, columns=lab), use_container_width=True)


tab1, tab2, tab3 = st.tabs(["1. Konstruksi S-Box (GA)", "2. Uji S-Box", "3. Enkripsi & Dekripsi Gambar"])

# ============================================================================
# TAB 1
# ============================================================================
with tab1:
    st.subheader("Konstruksi S-Box")
    st.markdown(
        "**Mode A – GA permutasi murni**: kromosom = permutasi 0..255 (otomatis *bijective* & *balance*). "
        "Crossover OX, mutasi swap, elitisme, + local search. Pada percobaan di sini NL berhenti di kisaran 98–102 "
        "(belum mencapai 112).  \n"
        "**Mode B – GA atas parameter afin**: kromosom = (A, B, c); S(x) = B·INV(A·x) ⊕ c dengan INV = invers GF(2⁸). "
        "NL/DU/LAP/DAP terjaga di level AES (NL 112), GA mengoptimalkan SAC, BIC-SAC, dan fixed point.")
    method = st.radio("Metode", ["Mode B – GA parameter afin", "Mode A – GA permutasi murni"], horizontal=True)

    c1, c2, c3 = st.columns(3)
    if method.startswith("Mode B"):
        pop = c1.number_input("Ukuran populasi", 20, 300, 100, step=10)
        gens = c2.number_input("Jumlah generasi", 10, 1000, 200, step=10)
        seed = c3.number_input("Seed", 0, 10_000, 42)
        st.caption("Perkiraan waktu: beberapa detik s.d. puluhan detik.")
    else:
        pop = c1.number_input("Ukuran populasi", 10, 200, 40, step=10)
        gens = c2.number_input("Jumlah generasi", 10, 1000, 150, step=10)
        seed = c3.number_input("Seed", 0, 10_000, 42)
        ls = st.slider("Langkah local search per generasi", 0, 300, 100, step=10)
        st.caption(f"Perkiraan waktu: ±{(pop * gens + gens * ls) * 1.5e-3 / 60:.1f} menit "
                   "(hasil terbaik untuk laporan sudah disiapkan di folder results/).")

    if st.button("▶ Jalankan GA", type="primary"):
        bar = st.progress(0.0, text="GA berjalan...")
        info = st.empty()

        def cb(g, rec):
            bar.progress(min(g / gens, 1.0), text=f"Generasi {g}/{gens}")
            if g % 5 == 0:
                info.write(f"cost {rec['cost']:.4f} | NL {rec['NL']} | DU {rec['DU']}")

        if method.startswith("Mode B"):
            res = run_affine_ga(AffineGAConfig(pop_size=int(pop), generations=int(gens), seed=int(seed)), cb)
            nm = "S-Box GA (afin)"
        else:
            res = run_ga(GAConfig(pop_size=int(pop), generations=int(gens), ls_steps=int(ls), seed=int(seed)), cb)
            nm = "S-Box GA (permutasi)"
        bar.progress(1.0, text="Selesai")
        st.session_state.sbox, st.session_state.sbox_name = res.best, nm
        st.session_state.history = res.history
        st.success(f"Selesai dalam {res.seconds:.1f} detik. S-Box aktif diperbarui.")

    with st.expander("Atau muat S-Box yang sudah ada"):
        files = sorted(RES.glob("sbox_*.json")) if RES.exists() else []
        if files:
            pick = st.selectbox("Hasil tersimpan", files, format_func=lambda p: p.name)
            if st.button("Muat hasil tersimpan"):
                d = load_sbox(pick)
                st.session_state.sbox, st.session_state.sbox_name = d["sbox"], d["name"]
                st.session_state.history = d.get("history", [])
                st.rerun()
        txt = st.text_area("Tempel S-Box sendiri (256 nilai heksadesimal, pisah koma/spasi/baris baru)", height=90)
        if st.button("Pakai S-Box tempelan"):
            try:
                st.session_state.sbox = parse_sbox_text(txt)
                st.session_state.sbox_name = "S-Box tempelan"
                st.session_state.history = []
                st.rerun()
            except Exception as e:
                st.error(str(e))
        if st.button("Pakai S-Box AES standar (pembanding)"):
            st.session_state.sbox, st.session_state.sbox_name = M.aes_sbox(), "AES standar"
            st.session_state.history = []
            st.rerun()

    st.markdown(f"### S-Box aktif: {st.session_state.sbox_name}")
    show_sbox_table(st.session_state.sbox)
    d1, d2 = st.columns(2)
    d1.download_button("Unduh CSV (hex 16×16)", sbox_to_csv(st.session_state.sbox), "sbox.csv", "text/csv")
    d2.download_button("Unduh JSON", json.dumps({"name": st.session_state.sbox_name,
                                                 "sbox": [int(v) for v in st.session_state.sbox]}),
                       "sbox.json", "application/json")
    if st.session_state.history:
        h = pd.DataFrame(st.session_state.history).set_index("gen")
        a, b = st.columns(2)
        a.markdown("**Konvergensi biaya**"); a.line_chart(h[["cost"]])
        b.markdown("**Nonlinearity terbaik per generasi**"); b.line_chart(h[["NL"]])

# ============================================================================
# TAB 2
# ============================================================================
with tab2:
    st.subheader(f"Uji kualitas S-Box: {st.session_state.sbox_name}")
    sb = st.session_state.sbox
    m, ma = M.evaluate(sb), M.evaluate(M.aes_sbox())

    ok1, ok2 = st.columns(2)
    (ok1.success if m["bijective"] else ok1.error)(f"Bijective: {'lolos' if m['bijective'] else 'GAGAL'} (semua 0–255 muncul tepat sekali)")
    (ok2.success if m["balanced"] else ok2.error)(f"Balance: {'lolos' if m['balanced'] else 'GAGAL'} — jumlah bit 1 per bit output: {m['balance_counts']}")

    def status(key, v):
        if key == "NL": return "✓" if v >= 112 else "✗ (< 112)"
        if key == "BIC_NL_min": return "✓" if v >= 112 else "✗ (< 112)"
        if key in ("SAC_mean", "BIC_SAC_mean"): return "✓" if abs(v - 0.5) <= 0.01 else "✗"
        if key in ("LAP", "DAP"): return "≤ AES" if v <= ma[key] + 1e-9 else "> AES"
        return ""

    rows = [("Nonlinearity (NL)", "NL", "≥ 112", "{:d}"), ("SAC (rata-rata)", "SAC_mean", "≈ 0.5", "{:.5f}"),
            ("BIC-NL (min)", "BIC_NL_min", "112", "{:d}"), ("BIC-SAC (rata-rata)", "BIC_SAC_mean", "≈ 0.5", "{:.5f}"),
            ("LAP", "LAP", "→ 0", "{:.5f}"), ("DAP", "DAP", "→ 0", "{:.5f}"),
            ("Differential uniformity", "DU", "4 (AES)", "{:d}"),
            ("Fixed point", "fixed_points", "0", "{:d}"), ("Opposite fixed point", "opposite_fixed_points", "0", "{:d}")]
    df = pd.DataFrame([{"Kriteria": r[0], "Target ideal": r[2], "S-Box aktif": r[3].format(m[r[1]]),
                        "AES standar": r[3].format(ma[r[1]]), "Status": status(r[1], m[r[1]])} for r in rows])
    st.table(df.set_index("Kriteria"))
    st.caption("Catatan: nilai terbaik yang diketahui untuk S-Box 8-bit bijektif adalah LAP ≈ 0.0625 dan "
               "DAP ≈ 0.0156 (setara AES); 'mendekati 0' berarti sedekat mungkin dengan nilai itu.")

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m["SAC_matrix"], cmap="viridis", vmin=0.4, vmax=0.6)
    ax.set_xlabel("bit output j"); ax.set_ylabel("bit input i"); ax.set_title("Matriks SAC (ideal 0.5)")
    fig.colorbar(im)
    left, right = st.columns([1, 2])
    left.pyplot(fig)
    right.markdown("**Rentang nilai**")
    right.write(f"SAC min/max: {m['SAC_min']:.4f} / {m['SAC_max']:.4f}  \n"
                f"BIC-SAC min/max: {m['BIC_SAC_min']:.4f} / {m['BIC_SAC_max']:.4f}  \n"
                f"BIC-NL rata-rata: {m['BIC_NL_mean']:.2f}")
    plt.close(fig)

# ============================================================================
# TAB 3
# ============================================================================
with tab3:
    st.subheader("AES dengan S-Box GA: enkripsi, dekripsi, dan uji kualitas")
    c1, c2, c3 = st.columns([2, 1, 1])
    key_txt = c1.text_input("Kunci (32 digit hex, atau teks bebas → di-hash SHA-256)", "2b7e151628aed2a6abf7158809cf4f3c")
    mode = c2.selectbox("Mode operasi", ["CBC", "ECB"], help="NPCR/UACI bermakna pada CBC. Pada ECB, mengubah 1 piksel "
                                                            "hanya memengaruhi 1 blok sehingga NPCR ≈ 0.")
    n_rand = c3.number_input("Posisi acak untuk NPCR/UACI", 0, 10, 2)
    compare = st.checkbox("Bandingkan dengan AES S-Box standar", value=True)

    ups = st.file_uploader("Unggah gambar (boleh banyak)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                           accept_multiple_files=True)
    use_ds = st.checkbox(f"Pakai semua gambar di folder `{DATASET}/`", value=not ups)
    items = [(u.name, IC.load_image(u.getvalue())) for u in ups]
    if use_ds:
        items += [(p.stem, IC.load_image(p)) for p in IC.list_dataset(DATASET)]

    if not items:
        st.info("Belum ada gambar. Unggah gambar, atau letakkan file di folder dataset/.")
    elif st.button("▶ Proses enkripsi & uji", type="primary"):
        key = key_to_bytes(key_txt, 128)
        aes_c = AES(key, st.session_state.sbox)
        aes_s = AES(key, M.aes_sbox()) if compare else None
        summary = []
        for name, img in items:
            with st.spinner(f"Memproses {name} ({img.shape[1]}×{img.shape[0]})..."):
                r, cimg, dimg = IC.analyze_image(img, aes_c, mode, DEFAULT_IV, int(n_rand))
                rs = IC.analyze_image(img, aes_s, mode, DEFAULT_IV, int(n_rand))[0] if aes_s else None
            st.markdown(f"#### {name}")
            a, b, c = st.columns(3)
            a.image(img, caption="Plain"); b.image(cimg, caption="Cipher"); c.image(dimg, caption="Hasil dekripsi")
            (st.success if r["decrypt_identical"] else st.error)(
                "Dekripsi identik dengan gambar asli (reversible)." if r["decrypt_identical"]
                else f"Dekripsi TIDAK identik (selisih maks {r['decrypt_max_abs_diff']}).")

            fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
            cols = ["r", "g", "b"] if img.ndim == 3 else ["k"]
            for h, cl in zip(IC.histograms(img), cols): ax[0].plot(h, color=cl, lw=.7)
            for h, cl in zip(IC.histograms(cimg), cols): ax[1].plot(h, color=cl, lw=.7)
            ax[0].set_title("Histogram plain"); ax[1].set_title("Histogram cipher")
            ax[1].set_ylim(0, max(h.max() for h in IC.histograms(cimg)) * 2)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            def line(label, rr):
                return {"Varian": label, "Entropy cipher (ideal ≈ 8)": f"{rr['entropy_cipher']:.5f}",
                        "Chi² (p-value)": f"{rr['chi2_cipher']:.1f} ({rr['chi2_pvalue']:.3f})",
                        "NPCR % (ideal ≈ 100)": f"{rr['NPCR_first']:.4f}", "UACI % (ideal ≈ 33.46)": f"{rr['UACI_first']:.4f}",
                        "Korelasi horizontal": f"{rr['corr_h_plain']:.3f} → {rr['corr_h_cipher']:.4f}"}
            tbl = [line(f"S-Box aktif ({st.session_state.sbox_name})", r)]
            if rs: tbl.append(line("AES standar", rs))
            st.table(pd.DataFrame(tbl).set_index("Varian"))
            st.caption("NPCR/UACI: 1 bit pada piksel pertama diubah, lalu plain dienkripsi ulang. "
                       f"Entropy plain = {r['entropy_plain']:.4f}. "
                       + (f"Rata-rata posisi acak: NPCR {r['NPCR_random']:.2f}%, UACI {r['UACI_random']:.2f}% "
                          "(pada CBC perubahan hanya merambat ke blok sesudahnya, jadi lebih rendah)." if n_rand else ""))
            buf = __import__("io").BytesIO(); Image.fromarray(cimg).save(buf, "PNG")
            st.download_button(f"Unduh cipher {name}", buf.getvalue(), f"{Path(name).stem}_cipher.png", "image/png",
                               key=f"dl_{name}")
            summary.append({"Gambar": name, "Entropy": round(r["entropy_cipher"], 5), "NPCR %": round(r["NPCR_first"], 4),
                            "UACI %": round(r["UACI_first"], 4), "Dekripsi identik": r["decrypt_identical"]})
        st.markdown("### Ringkasan semua gambar")
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
