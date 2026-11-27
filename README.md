# Proyek S-Box dengan Algoritma Genetika → AES Modifikasi → Enkripsi Gambar

Kriptografi TI – proyek S-Box. Alur: bangun S-Box (GA) → uji 6+ kriteria → ganti S-Box AES →
enkripsi & dekripsi gambar → uji Entropy / Histogram / NPCR / UACI.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py              # aplikasi (3 tab)
python run_pipeline.py            # semua tahap tanpa Streamlit -> folder results/
python test_all.py                # 23 uji regresi (±1 detik)
python make_report.py             # draf laporan PDF dari hasil di results/
```

Gambar dataset (cameraman, lena_color_512, livingroom, mandril_color, pirate, woman_blonde, woman_darkhair)
cukup disalin ke folder `dataset/`. Bila folder kosong, `run_pipeline.py` membuat 3 gambar sintetis sebagai pengganti
sementara. Setelah gambar asli masuk: hapus `sintetis_*.png`, jalankan `python run_pipeline.py --skip-ga`
lalu `python make_report.py`.

## Isi

| File | Fungsi |
|---|---|
| `sbox_metrics.py` | Balance, bijective, NL, SAC, BIC-NL, BIC-SAC, LAP, DAP, DU, fixed point |
| `sbox_ga.py` | GA mode A (permutasi) dan mode B (parameter afin) |
| `aes_custom.py` | AES-128/192/256 dengan S-Box dapat diganti; ECB & CBC |
| `image_crypto.py` | Enkripsi/dekripsi gambar, Entropy, histogram, chi-square, NPCR, UACI, korelasi |
| `sbox_io.py` | Simpan/muat S-Box (JSON/CSV) |
| `app.py` | Aplikasi Streamlit |
| `run_pipeline.py`, `make_report.py`, `test_all.py`, `make_sample_images.py` | Skrip pendukung |
| `results/` | S-Box hasil GA (JSON), tabel metrik (CSV), grafik konvergensi, gambar cipher/dekripsi/histogram |

## Dua mode GA

**Mode A – permutasi murni.** Kromosom = permutasi 0..255 (otomatis bijective & balance). Seleksi turnamen,
order crossover (OX1), mutasi swap, elitisme, local search pada elit. Fungsi biaya:
`mean(||W|-12|^6)/16^6 + 1.0*mean(DDT^3) + 20*mean|SAC-0.5|`.
Hasil run tersimpan: NL 100, DU 8, LAP 0.109, DAP 0.031 → **belum memenuhi NL ≥ 112**.

**Mode B – parameter afin.** Kromosom = (A, B, c); `S(x) = B·INV(A·x) ⊕ c`, INV = invers GF(2⁸).
NL/DU/LAP/DAP setara AES (NL 112, DU 4, LAP 0.0625, DAP 0.0156); GA mengoptimalkan SAC, BIC-SAC, fixed point.
Karena strukturnya kelas ekuivalen AES, tanyakan ke Pak Alam apakah ini dianggap "konstruksi dengan ML".

## Catatan penting

- **NPCR/UACI** dihitung dengan membalik 1 bit piksel pertama lalu mengenkripsi ulang. Pada **ECB** hasilnya ≈ 0%
  (1 piksel hanya memengaruhi 1 blok) — itu sifat mode, bukan S-Box. Pakai **CBC** (default).
- Implementasi AES diverifikasi dengan test vector FIPS-197 dan dicocokkan dengan library `cryptography`
  (ECB/CBC, kunci 128/192/256) saat memakai S-Box standar.
- Kunci: 32 digit hex, atau teks bebas (di-hash SHA-256 → 128 bit). IV default tetap (`00 01 … 0F`); untuk pemakaian
  nyata, IV seharusnya acak dan disimpan bersama cipher.
- Padding nol dipakai bila ukuran data bukan kelipatan 16 byte (panjang asli disimpan oleh pemanggil).
- Untuk deploy Streamlit Community Cloud: unggah folder ini (termasuk `results/` dan `dataset/`) ke GitHub.
