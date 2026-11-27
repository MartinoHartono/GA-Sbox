"""
Membuat DRAF laporan PDF dari hasil di results/ (angka diambil otomatis dari file hasil).
Jalankan setelah run_pipeline.py:   python make_report.py   ->  Draf_Laporan_SBox_GA.pdf
Isi nama/NIM langsung di PDF atau ubah konstanta STUDENT di bawah.
"""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import sbox_metrics as M
from sbox_io import load_sbox

STUDENT = "[Nama Mahasiswa] / [NIM] - Kriptografi TI Rombel 2"
RES = Path("results")
OUT = "Draf_Laporan_SBox_GA.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], alignment=TA_JUSTIFY, fontSize=10, leading=14)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=ss["Title"], fontSize=17, leading=21)

P = lambda t, s=body: Paragraph(t, s)


def tbl(data, widths=None, header=True, fs=8.5, align_left_first=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER")]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def sbox_table(sbox):
    hexm = [[f"{int(v):02X}" for v in row] for row in sbox.reshape(16, 16)]
    data = [[""] + [f"{i:X}" for i in range(16)]] + [[f"{i:X}"] + hexm[i] for i in range(16)]
    t = Table(data, colWidths=[0.7 * cm] + [0.95 * cm] * 16, rowHeights=0.5 * cm)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DDE6F0")),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def main():
    A, B = load_sbox(RES / "sbox_ga_permutasi.json"), load_sbox(RES / "sbox_ga_afin.json")
    cand = {"AES standar": M.aes_sbox(), "GA Mode A (permutasi)": A["sbox"], "GA Mode B (afin)": B["sbox"]}
    mt = {k: M.evaluate(v) for k, v in cand.items()}
    names = list(cand)

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Draf Laporan S-Box GA", author=STUDENT)
    s = []
    s += [P("DRAF LAPORAN PROYEK", title),
          P("Konstruksi S-Box dengan Algoritma Genetika dan Implementasinya pada AES untuk Enkripsi Citra",
            ParagraphStyle("t2", parent=title, fontSize=13, leading=17)),
          Spacer(1, 4), P(STUDENT, ParagraphStyle("c", parent=body, alignment=1)),
          P("<i>Draf untuk review dosen - hasil sementara dan arah metode; belum final.</i>",
            ParagraphStyle("c2", parent=small, alignment=1)), Spacer(1, 8)]

    # 1 ------------------------------------------------------------------
    s += [P("1. Tujuan dan Alur Proyek", h1),
          P("Proyek membangun S-Box 16x16 (nilai 0-255) dengan pendekatan machine learning (Algoritma Genetika), "
            "mengganti S-Box AES dengan S-Box tersebut, mengenkripsi dan mendekripsi citra, lalu menguji kualitas "
            "S-Box dan hasil enkripsi. Brute force tidak mungkin karena ruang pencarian 256! permutasi."),
          Spacer(1, 4),
          tbl([["Tahap", "Keluaran / pengujian"],
               ["1. Konstruksi S-Box (GA)", P("S-Box bijective dan balance (syarat awal)", small)],
               ["2. Uji S-Box", P("NL, SAC, BIC-NL, BIC-SAC, LAP, DAP (+ DU, fixed point)", small)],
               ["3. Substitusi pada AES-128", P("SubBytes dan key expansion (SubWord) memakai S-Box GA; "
                                                 "InvSubBytes memakai inversnya", small)],
               ["4. Enkripsi citra", P("Entropy, histogram (+ chi-square), NPCR, UACI, korelasi piksel", small)],
               ["5. Dekripsi citra", P("Bukti reversibel: citra hasil dekripsi identik dengan citra asli", small)]],
              [5 * cm, 12 * cm])]

    # 2 ------------------------------------------------------------------
    s += [P("2. Rancangan Metode Genetika", h1),
          P("Dua varian GA dibandingkan secara terbuka:"),
          P("<b>Mode A - GA permutasi murni.</b> Kromosom adalah permutasi bilangan 0-255 sehingga setiap individu "
            "otomatis bijective dan balance. Operator: seleksi turnamen (k=3), order crossover (OX1), mutasi swap "
            "(1-3 pasang), elitisme (2 individu), serta local search (hill-climbing swap) pada individu terbaik tiap generasi "
            "(pendekatan memetic). Fungsi biaya yang diminimalkan:"),
          P("cost = mean(| |W(b,a)| - X |<super>R</super>) / 16<super>R</super> + w<sub>dp</sub> * mean(DDT<super>3</super>) "
            "+ w<sub>sac</sub> * mean(|SAC - 0.5|),  dengan X=12, R=6, w<sub>dp</sub>=1, w<sub>sac</sub>=20.",
            ParagraphStyle("f", parent=body, alignment=1)),
          P("W adalah spektrum Walsh-Hadamard seluruh kombinasi linear bit output; suku pertama menekan puncak spektrum "
            "(menaikkan NL dan menurunkan LAP), suku kedua menekan nilai besar pada tabel diferensial (menurunkan DAP), "
            "suku ketiga menjaga SAC mendekati 0,5."),
          P("<b>Mode B - GA atas parameter transformasi afin.</b> S(x) = B . INV(A . x) XOR c, dengan INV adalah invers "
            "pada GF(2<super>8</super>) (polinomial 0x11B), A dan B matriks 8x8 invertibel atas GF(2), c konstanta 8-bit. "
            "Kromosom = (A, B, c). Mutasi memakai operasi baris elementer (menjaga invertibilitas) dan pembalikan bit "
            "konstanta; crossover memilih baris dari kedua induk dengan pemeriksaan invertibilitas. Transformasi afin "
            "menjaga NL, DU, LAP, dan DAP, sedangkan SAC, BIC-SAC, dan fixed point berubah - itulah yang dioptimalkan GA. "
            "Biaya = penyimpangan SAC dan BIC-SAC dari 0,5 + penalti fixed point."),
          P("Parameter run yang dilaporkan: Mode A populasi %d, %d generasi, %d langkah local search, seed %d; "
            "Mode B populasi %d, %d generasi, seed %d." % (
                A["config"]["pop_size"], A["config"]["generations"], A["config"]["ls_steps"], A["config"]["seed"],
                B["config"]["pop_size"], B["config"]["generations"], B["config"]["seed"]), small)]

    # 3 ------------------------------------------------------------------
    s += [PageBreak(), P("3. Hasil Uji S-Box", h1)]
    fmt = lambda k, v: f"{v:.5f}" if isinstance(v, float) else str(v)
    rows = [["Kriteria", "Target"] + names]
    for label, k, tg in [("Bijective", "bijective", "Ya"), ("Balance (128:128)", "balanced", "Ya"),
                         ("Nonlinearity (NL)", "NL", ">= 112"), ("SAC (rata-rata)", "SAC_mean", "~ 0.5"),
                         ("BIC-NL (min)", "BIC_NL_min", "112"), ("BIC-SAC (rata-rata)", "BIC_SAC_mean", "~ 0.5"),
                         ("LAP", "LAP", "-> 0"), ("DAP", "DAP", "-> 0"), ("Differential uniformity", "DU", "4"),
                         ("Fixed point", "fixed_points", "0")]:
        rows.append([label, tg] + [("Ya" if mt[n][k] is True else "Tidak" if mt[n][k] is False else fmt(k, mt[n][k]))
                                   for n in names])
    s += [tbl(rows, [4.2 * cm, 2 * cm, 3.4 * cm, 4 * cm, 3.4 * cm]), Spacer(1, 6)]
    na, nb = mt[names[1]], mt[names[2]]
    s += [P(f"<b>Pembacaan hasil.</b> Mode B mencapai NL = {nb['NL']}, BIC-NL = {nb['BIC_NL_min']}, DU = {nb['DU']}, "
            f"LAP = {nb['LAP']:.5f}, DAP = {nb['DAP']:.5f}, dan BIC-SAC = {nb['BIC_SAC_mean']:.5f} (AES standar: "
            f"{mt[names[0]]['BIC_SAC_mean']:.5f}). Nilai NL/DU/LAP/DAP sama dengan AES karena Mode B berada pada kelas "
            f"ekuivalen afin invers GF(2<super>8</super>); LAP dan DAP tidak dapat lebih baik dari AES pada kelas ini. "
            f"Mode A (permutasi murni) berhenti pada NL = {na['NL']}, DU = {na['DU']}, LAP = {na['LAP']:.5f}, "
            f"DAP = {na['DAP']:.5f}, sehingga <b>belum memenuhi target NL >= 112</b>. Pada eksperimen ini pencarian "
            f"permutasi murni tertahan di sekitar NL 100 (kurva pada Gambar 1); [tambahkan sitasi pembanding dari "
            f"literatur sebelum laporan final]."),
          Spacer(1, 4),
          P("S-Box GA Mode B (heksadesimal; baris = nibble tinggi, kolom = nibble rendah):", h2), sbox_table(B["sbox"]),
          Spacer(1, 6)]
    if (RES / "konvergensi_ga.png").exists():
        s += [KeepTogether([Image(str(RES / "konvergensi_ga.png"), width=14 * cm, height=5.04 * cm),
                            P("Gambar 1. Konvergensi biaya dan NL individu terbaik pada kedua mode GA.", small)])]

    # 4 ------------------------------------------------------------------
    s += [Spacer(1, 6), P("4. Hasil Enkripsi dan Dekripsi Citra (AES-128, mode CBC)", h1)]
    gp = RES / "metrik_gambar.csv"
    if gp.exists():
        with open(gp, encoding="utf-8") as f:
            r = list(csv.reader(f))
        hdr = r[0]
        ix = {h: i for i, h in enumerate(hdr)}
        data = [["Gambar", "S-Box", "Entropy", "Chi2 (p)", "NPCR %", "UACI %", "Dekripsi identik"]]
        for row in r[1:]:
            data.append([row[ix["gambar"]], row[ix["S-Box"]], row[ix["entropy_cipher"]],
                         f"{row[ix['chi2']]} ({row[ix['chi2_p']]})", row[ix["NPCR_first_%"]], row[ix["UACI_first_%"]],
                         "Ya" if row[ix["dekripsi_identik"]] == "True" else "TIDAK"])
        s += [tbl(data, [2.6 * cm, 4 * cm, 1.9 * cm, 2.6 * cm, 1.8 * cm, 1.8 * cm, 2.3 * cm], fs=7.5), Spacer(1, 4),
              P("NPCR dan UACI dihitung dengan membalik 1 bit pada piksel pertama, lalu mengenkripsi ulang dan "
                "membandingkan dua cipher. Nilai ideal: Entropy ~ 8, NPCR ~ 100%, UACI ~ 33,46%. Pada mode ECB, "
                "perubahan satu piksel hanya memengaruhi satu blok (16 byte) sehingga NPCR mendekati 0; uji difusi "
                "karena itu dilakukan pada mode CBC.", small), Spacer(1, 6)]
    first_hist = sorted((RES / "gambar").glob("*_histogram.png"))
    if first_hist:
        s += [KeepTogether([Image(str(first_hist[0]), width=13 * cm, height=8.7 * cm),
                            P(f"Gambar 2. Contoh plain, cipher, dan histogram ({first_hist[0].stem.replace('_histogram', '')}). "
                              "Histogram cipher mendekati seragam.", small)])]

    # 5 ------------------------------------------------------------------
    s += [P("5. Keterbatasan Draf Ini dan Rencana Lanjutan", h1),
          P("- <b>Dataset:</b> hasil di atas memakai gambar sintetis sebagai pengganti sementara. Tujuh citra dataset "
            "(cameraman, lena_color_512, livingroom, mandril_color, pirate, woman_blonde, woman_darkhair) akan "
            "dimasukkan ke folder dataset/ dan seluruh tabel dibuat ulang otomatis."),
          P("- <b>Target NL >= 112:</b> hanya dicapai Mode B, yang berbasis invers GF(2<super>8</super>) dengan GA atas "
            "transformasi afin; Mode A (GA permutasi murni) berhenti pada NL " + str(na["NL"]) + ". Arah yang ingin "
            "dikonsultasikan: apakah Mode B dapat diterima sebagai 'konstruksi dengan ML', atau Mode A perlu diperkuat "
            "(mis. dikombinasikan dengan modifikasi titik/matriks seperti pada paper acuan)."),
          P("- <b>LAP dan DAP:</b> nilai terbaik yang diketahui untuk S-Box 8-bit bijektif adalah sekitar 0,0625 (LAP) dan "
            "0,0156 (DAP), yaitu nilai AES; 'mendekati 0' diartikan sedekat mungkin ke nilai tersebut."),
          P("- <b>Rencana:</b> uji tambahan (transparency order, korelasi piksel arah vertikal/diagonal), ulang eksperimen "
            "dengan beberapa seed (laporkan rata-rata dan simpangan baku), finalisasi aplikasi Streamlit, dan penyusunan "
            "laporan akhir.")]
    doc.build(s)
    print("Dibuat:", OUT)


if __name__ == "__main__":
    main()
