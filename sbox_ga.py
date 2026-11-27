"""
Konstruksi S-Box 8x8 dengan Algoritma Genetika (GA).

Representasi   : kromosom = permutasi 0..255  -> otomatis BIJECTIVE (dan BALANCE).
Seleksi        : turnamen (ukuran k).
Crossover      : Order Crossover (OX1) -> anak selalu permutasi valid.
Mutasi         : tukar (swap) beberapa pasang posisi.
Elitisme       : individu terbaik dibawa ke generasi berikutnya.
Memetic step   : local search (swap-hill-climbing) pada individu elit tiap generasi.
Fungsi biaya   : (diminimalkan)
      cost = mean( | |W(b,a)| - X |^R )  / norm    <- mendorong NL tinggi & LAP rendah
           + w_dp * mean( DDT^3 )                  <- mendorong DAP rendah
           + w_sac * (penyimpangan SAC dari 0.5)    <- mendorong SAC ≈ 0.5
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

import sbox_metrics as M

_IDX = np.arange(256)


# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
@dataclass
class GAConfig:
    pop_size: int = 80
    generations: int = 400
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.6          # peluang sebuah anak dimutasi
    max_swaps: int = 3                  # banyaknya swap per mutasi (1..max_swaps)
    elite: int = 2
    ls_steps: int = 150                 # percobaan swap local search pada elit / generasi
    # bobot fungsi biaya (R besar menekan puncak spektrum Walsh -> NL naik)
    walsh_X: float = 12.0
    walsh_R: float = 6.0
    w_dp: float = 1.0
    w_sac: float = 20.0
    seed: int | None = 42

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Fungsi biaya
# --------------------------------------------------------------------------
def cost(s: np.ndarray, cfg: GAConfig) -> float:
    W = np.abs(M.walsh_all(s)[1:])                         # (255, 256) float32
    d = np.abs(W - np.float32(cfg.walsh_X))
    p = d.copy()
    for _ in range(int(cfg.walsh_R) - 1):                  # pangkat bulat via perkalian (jauh lebih cepat)
        p *= d
    c_nl = float(p.mean()) / (16.0 ** int(cfg.walsh_R))
    T = M.ddt(s)[1:].astype(np.float32)
    c_dp = float((T * T * T).mean())
    sac = M.sac_matrix(s)
    c_sac = np.mean(np.abs(sac - 0.5))
    return float(c_nl + cfg.w_dp * c_dp + cfg.w_sac * c_sac)


# --------------------------------------------------------------------------
# Operator GA
# --------------------------------------------------------------------------
def order_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OX1: salin potongan p1, sisanya diisi urutan p2."""
    n = p1.size
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=p1.dtype)
    child[a:b + 1] = p1[a:b + 1]
    used = np.zeros(n, dtype=bool)
    used[p1[a:b + 1]] = True
    rest = p2[~used[p2]]
    pos = np.r_[np.arange(b + 1, n), np.arange(0, a)]
    child[pos] = rest
    return child


def mutate(s: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    s = s.copy()
    for _ in range(int(rng.integers(1, cfg.max_swaps + 1))):
        i, j = rng.choice(256, size=2, replace=False)
        s[i], s[j] = s[j], s[i]
    return s


def local_search(s: np.ndarray, c: float, steps: int, cfg: GAConfig,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Hill-climbing: terima swap acak hanya bila biaya turun."""
    s = s.copy()
    for _ in range(steps):
        i, j = rng.choice(256, size=2, replace=False)
        s[i], s[j] = s[j], s[i]
        c2 = cost(s, cfg)
        if c2 < c:
            c = c2
        else:
            s[i], s[j] = s[j], s[i]
    return s, c


def tournament(pop, costs, k, rng):
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmin(costs[idx])]]


# --------------------------------------------------------------------------
# Loop utama
# --------------------------------------------------------------------------
@dataclass
class GAResult:
    best: np.ndarray
    best_cost: float
    history: list = field(default_factory=list)   # dict per generasi
    config: dict = field(default_factory=dict)
    seconds: float = 0.0


def run_ga(cfg: GAConfig, progress: Callable[[int, dict], None] | None = None,
           init_pop: list[np.ndarray] | None = None) -> GAResult:
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    pop = [rng.permutation(256) for _ in range(cfg.pop_size)]
    if init_pop:
        for i, p in enumerate(init_pop[:cfg.pop_size]):
            pop[i] = np.asarray(p, dtype=np.int64).copy()
    costs = np.array([cost(p, cfg) for p in pop])

    history = []
    for gen in range(cfg.generations + 1):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        costs = costs[order]

        # memetic: poles elit terbaik
        pop[0], costs[0] = local_search(pop[0], costs[0], cfg.ls_steps, cfg, rng)

        best = pop[0]
        W = M.walsh_all(best)
        rec = {
            "gen": gen,
            "cost": float(costs[0]),
            "mean_cost": float(costs.mean()),
            "NL": M.nonlinearity(best, W),
            "DU": M.differential_uniformity(best),
        }
        history.append(rec)
        if progress:
            progress(gen, rec)
        if gen == cfg.generations:
            break

        # generasi baru
        new_pop = [p.copy() for p in pop[:cfg.elite]]
        new_cost = list(costs[:cfg.elite])
        while len(new_pop) < cfg.pop_size:
            p1 = tournament(pop, costs, cfg.tournament_k, rng)
            p2 = tournament(pop, costs, cfg.tournament_k, rng)
            child = order_crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1.copy()
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, cfg, rng)
            new_pop.append(child)
            new_cost.append(cost(child, cfg))
        pop, costs = new_pop, np.array(new_cost)

    order = np.argsort(costs)
    best = pop[order[0]]
    return GAResult(best=best.astype(np.int64), best_cost=float(costs[order[0]]),
                    history=history, config=cfg.to_dict(), seconds=time.time() - t0)


# ==========================================================================
# MODE B: GA atas parameter transformasi afin pada invers GF(2^8)
# ==========================================================================
# S(x) = B · INV( A · x )  XOR  c        (A, B: matriks 8x8 invertibel atas GF(2); c: konstanta)
# Transformasi afin menjaga NL, DU, LAP, DAP (setara dengan invers GF(2^8)),
# sedangkan SAC / BIC-SAC / fixed point berubah -> inilah yang dioptimalkan GA.
# Kromosom = (A, B, c)  ;  A,B disimpan sebagai 8 baris (byte).
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


def _build_inverse() -> np.ndarray:
    inv = np.zeros(256, dtype=np.int64)
    for x in range(1, 256):
        for y in range(1, 256):
            if _gmul(x, y) == 1:
                inv[x] = y
                break
    return inv


GF_INV = _build_inverse()


def mat_apply(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = M·x untuk semua x; rows[k] = baris ke-k (byte) -> y_k = parity(rows[k] & x)."""
    bits = M.PARITY[np.bitwise_and.outer(rows, x)].astype(np.int64)   # (8, len(x))
    return (bits << np.arange(8)[:, None]).sum(axis=0)


def mat_invertible(rows) -> bool:
    r = [int(v) for v in rows]
    for col in range(8):
        piv = next((i for i in range(col, 8) if (r[i] >> col) & 1), None)
        if piv is None:
            return False
        r[col], r[piv] = r[piv], r[col]
        for i in range(8):
            if i != col and (r[i] >> col) & 1:
                r[i] ^= r[col]
    return True


def random_invertible(rng: np.random.Generator) -> np.ndarray:
    while True:
        rows = rng.integers(1, 256, size=8)
        if mat_invertible(rows):
            return rows.astype(np.int64)


def build_affine_sbox(A: np.ndarray, B: np.ndarray, c: int) -> np.ndarray:
    xs = np.arange(256)
    return mat_apply(B, GF_INV[mat_apply(A, xs)]) ^ int(c)


@dataclass
class AffineGAConfig:
    pop_size: int = 60
    generations: int = 150
    tournament_k: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.8
    elite: int = 2
    # bobot biaya
    w_sac: float = 1.0
    w_bic: float = 1.0
    w_fixed: float = 0.05
    seed: int | None = 42

    def to_dict(self):
        return asdict(self)


def affine_cost(s: np.ndarray, cfg: AffineGAConfig) -> float:
    sac = M.sac_matrix(s)
    bic = M.bic_sac(s)
    fp, ofp = M.fixed_points(s)
    return float(cfg.w_sac * (np.abs(sac - 0.5).mean() + np.abs(sac - 0.5).max())
                 + cfg.w_bic * (abs(bic["mean"] - 0.5) + max(bic["max"] - 0.5, 0.5 - bic["min"]))
                 + cfg.w_fixed * (fp + ofp))


def _affine_crossover(p1, p2, rng):
    A1, B1, c1 = p1
    A2, B2, c2 = p2
    for _ in range(20):
        mask = rng.random(8) < 0.5
        A = np.where(mask, A1, A2)
        if mat_invertible(A):
            break
    else:
        A = A1.copy()
    for _ in range(20):
        mask = rng.random(8) < 0.5
        B = np.where(mask, B1, B2)
        if mat_invertible(B):
            break
    else:
        B = B1.copy()
    c = int(c1 if rng.random() < 0.5 else c2)
    return A.astype(np.int64), B.astype(np.int64), c


def _affine_mutate(ind, rng):
    A, B, c = ind[0].copy(), ind[1].copy(), int(ind[2])
    r = rng.random()
    if r < 0.4:                       # operasi baris elementer (menjaga invertibilitas)
        i, j = rng.choice(8, size=2, replace=False)
        A[i] ^= A[j]
    elif r < 0.8:
        i, j = rng.choice(8, size=2, replace=False)
        B[i] ^= B[j]
    else:
        c ^= 1 << int(rng.integers(0, 8))
    return A, B, c


def run_affine_ga(cfg: AffineGAConfig, progress: Callable[[int, dict], None] | None = None) -> GAResult:
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()
    pop = [(random_invertible(rng), random_invertible(rng), int(rng.integers(0, 256)))
           for _ in range(cfg.pop_size)]
    sboxes = [build_affine_sbox(*p) for p in pop]
    costs = np.array([affine_cost(s, cfg) for s in sboxes])

    history = []
    for gen in range(cfg.generations + 1):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        sboxes = [sboxes[i] for i in order]
        costs = costs[order]
        rec = {"gen": gen, "cost": float(costs[0]), "mean_cost": float(costs.mean()),
               "NL": M.nonlinearity(sboxes[0]), "DU": M.differential_uniformity(sboxes[0])}
        history.append(rec)
        if progress:
            progress(gen, rec)
        if gen == cfg.generations:
            break

        new_pop, new_sb, new_c = pop[:cfg.elite], sboxes[:cfg.elite], list(costs[:cfg.elite])
        while len(new_pop) < cfg.pop_size:
            ia = rng.choice(len(pop), size=cfg.tournament_k, replace=False)
            ib = rng.choice(len(pop), size=cfg.tournament_k, replace=False)
            p1, p2 = pop[ia[np.argmin(costs[ia])]], pop[ib[np.argmin(costs[ib])]]
            child = _affine_crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1
            if rng.random() < cfg.mutation_rate:
                child = _affine_mutate(child, rng)
            s = build_affine_sbox(*child)
            new_pop.append(child)
            new_sb.append(s)
            new_c.append(affine_cost(s, cfg))
        pop, sboxes, costs = new_pop, new_sb, np.array(new_c)

    i = int(np.argmin(costs))
    return GAResult(best=sboxes[i].astype(np.int64), best_cost=float(costs[i]), history=history,
                    config=cfg.to_dict(), seconds=time.time() - t0)
