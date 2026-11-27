"""Simpan / muat S-Box (JSON, CSV) beserta metriknya."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def sbox_matrix_hex(sbox) -> list[list[str]]:
    m = np.asarray(sbox).reshape(16, 16)
    return [[f"{int(v):02X}" for v in row] for row in m]


def sbox_to_csv(sbox) -> str:
    """CSV 16x16 berisi nilai heksadesimal (tanpa header)."""
    return "\n".join(",".join(r) for r in sbox_matrix_hex(sbox))


def save_sbox(path, name: str, method: str, sbox, metrics: dict | None = None,
              config: dict | None = None, seconds: float | None = None,
              history: list | None = None):
    data = {
        "name": name,
        "method": method,
        "sbox": [int(v) for v in np.asarray(sbox).reshape(-1)],
        "matrix_hex": sbox_matrix_hex(sbox),
        "metrics": metrics or {},
        "config": config or {},
        "seconds": seconds,
        "history": history or [],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_clean(data), indent=1), encoding="utf-8")


def load_sbox(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["sbox"] = np.array(d["sbox"], dtype=np.int64)
    return d


def parse_sbox_text(text: str) -> np.ndarray:
    """Baca S-Box dari teks CSV/spasi: 256 angka (heks tanpa awalan, atau desimal bila ada angka >255 / titik)."""
    toks = [t for t in text.replace(",", " ").replace(";", " ").split() if t]
    if len(toks) != 256:
        raise ValueError(f"Butuh 256 nilai, ditemukan {len(toks)}.")
    vals = []
    for t in toks:
        t = t.lower().removeprefix("0x")
        vals.append(int(t, 16))
    arr = np.array(vals, dtype=np.int64)
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError("Nilai di luar 0..255.")
    return arr
