"""
AES (128/192/256-bit) dengan S-Box yang dapat diganti.

- SubBytes & SubWord (key expansion) memakai S-Box kustom; InvSubBytes memakai inversnya.
- ECB  : divektorisasi numpy (enkripsi & dekripsi cepat).
- CBC  : enkripsi berantai (tabel-T, python murni), dekripsi divektorisasi.
Dengan S-Box standar, hasilnya identik dengan AES FIPS-197 (diverifikasi di test_all.py).
"""
from __future__ import annotations

import numpy as np

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D]


def _xtime(a: int) -> int:
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def _mul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


MUL = {k: np.array([_mul(x, k) for x in range(256)], dtype=np.uint8) for k in (2, 3, 9, 11, 13, 14)}

# ShiftRows pada urutan byte kolom-mayor: idx = r + 4c
_SR = np.array([(i % 4) + 4 * (((i // 4) + (i % 4)) % 4) for i in range(16)])
_ISR = np.empty(16, dtype=int)
_ISR[_SR] = np.arange(16)


def key_to_bytes(key: str | bytes, bits: int = 128) -> bytes:
    """Kunci dari teks/hex -> bytes. Hex valid dipakai apa adanya; selain itu di-hash SHA-256."""
    n = bits // 8
    if isinstance(key, str):
        k = key.strip()
        try:
            raw = bytes.fromhex(k)
            if len(raw) == n:
                return raw
        except ValueError:
            pass
        import hashlib
        return hashlib.sha256(k.encode()).digest()[:n]
    if len(key) != n:
        raise ValueError(f"Kunci harus {n} byte.")
    return key


class AES:
    def __init__(self, key: bytes, sbox: np.ndarray | None = None):
        if len(key) not in (16, 24, 32):
            raise ValueError("Panjang kunci harus 16/24/32 byte.")
        if sbox is None:
            from sbox_metrics import aes_sbox
            sbox = aes_sbox()
        sbox = np.asarray(sbox, dtype=np.int64).reshape(-1)
        if len(np.unique(sbox)) != 256 or sbox.min() < 0 or sbox.max() > 255:
            raise ValueError("S-Box harus permutasi 0..255 (bijective).")
        self.sbox = sbox.astype(np.uint8)
        self.inv_sbox = np.empty(256, dtype=np.uint8)
        self.inv_sbox[self.sbox] = np.arange(256, dtype=np.uint8)
        self.nk = len(key) // 4
        self.nr = self.nk + 6
        self.round_keys = self._expand_key(key)                 # (nr+1, 16) uint8
        self._build_ttables()

    # ----------------------------------------------------------------- kunci
    def _expand_key(self, key: bytes) -> np.ndarray:
        S = self.sbox
        nk, nr = self.nk, self.nr
        w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
        for i in range(nk, 4 * (nr + 1)):
            t = list(w[i - 1])
            if i % nk == 0:
                t = t[1:] + t[:1]
                t = [int(S[b]) for b in t]
                t[0] ^= RCON[i // nk - 1]
            elif nk > 6 and i % nk == 4:
                t = [int(S[b]) for b in t]
            w.append([a ^ b for a, b in zip(w[i - nk], t)])
        flat = np.array(w, dtype=np.uint8).reshape(nr + 1, 16)
        return flat

    # ------------------------------------------------- ECB/dekripsi (numpy)
    def encrypt_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """blocks: (N,16) uint8 -> (N,16) uint8 (mode ECB)."""
        st = blocks ^ self.round_keys[0]
        for r in range(1, self.nr):
            st = self.sbox[st][:, _SR]
            st = self._mix_fwd(st)
            st = st ^ self.round_keys[r]
        st = self.sbox[st][:, _SR]
        return st ^ self.round_keys[self.nr]

    def _mix_fwd(self, st):
        s = st.reshape(-1, 4, 4)
        a0, a1, a2, a3 = s[:, :, 0], s[:, :, 1], s[:, :, 2], s[:, :, 3]
        m2, m3 = MUL[2], MUL[3]
        out = np.empty_like(s)
        out[:, :, 0] = m2[a0] ^ m3[a1] ^ a2 ^ a3
        out[:, :, 1] = a0 ^ m2[a1] ^ m3[a2] ^ a3
        out[:, :, 2] = a0 ^ a1 ^ m2[a2] ^ m3[a3]
        out[:, :, 3] = m3[a0] ^ a1 ^ a2 ^ m2[a3]
        return out.reshape(-1, 16)

    def _mix_inv(self, st):
        s = st.reshape(-1, 4, 4)
        a0, a1, a2, a3 = s[:, :, 0], s[:, :, 1], s[:, :, 2], s[:, :, 3]
        m9, m11, m13, m14 = MUL[9], MUL[11], MUL[13], MUL[14]
        out = np.empty_like(s)
        out[:, :, 0] = m14[a0] ^ m11[a1] ^ m13[a2] ^ m9[a3]
        out[:, :, 1] = m9[a0] ^ m14[a1] ^ m11[a2] ^ m13[a3]
        out[:, :, 2] = m13[a0] ^ m9[a1] ^ m14[a2] ^ m11[a3]
        out[:, :, 3] = m11[a0] ^ m13[a1] ^ m9[a2] ^ m14[a3]
        return out.reshape(-1, 16)

    def decrypt_blocks(self, blocks: np.ndarray) -> np.ndarray:
        st = blocks ^ self.round_keys[self.nr]
        for r in range(self.nr - 1, 0, -1):
            st = self.inv_sbox[st[:, _ISR]]
            st = st ^ self.round_keys[r]
            st = self._mix_inv(st)
        st = self.inv_sbox[st[:, _ISR]]
        return st ^ self.round_keys[0]

    # ------------------------------------------- CBC enkripsi (tabel-T, python)
    def _build_ttables(self):
        S = self.sbox.astype(np.int64)
        m2 = MUL[2].astype(np.int64)
        m3 = MUL[3].astype(np.int64)
        te0 = (m2[S] << 24) | (S << 16) | (S << 8) | m3[S]
        ror = lambda v, n: ((v >> n) | (v << (32 - n))) & 0xFFFFFFFF
        self._te0 = te0.tolist()
        self._te1 = [ror(v, 8) for v in self._te0]
        self._te2 = [ror(v, 16) for v in self._te0]
        self._te3 = [ror(v, 24) for v in self._te0]
        self._s = self.sbox.astype(np.int64).tolist()
        rk = self.round_keys.reshape(-1, 4, 4).astype(np.int64)         # (rounds+1, 4 kata, 4 byte)
        self._rkw = [[int((w[0] << 24) | (w[1] << 16) | (w[2] << 8) | w[3]) for w in r] for r in rk]

    def _enc_words(self, s0, s1, s2, s3):
        te0, te1, te2, te3, S, rkw = self._te0, self._te1, self._te2, self._te3, self._s, self._rkw
        k = rkw[0]
        s0 ^= k[0]; s1 ^= k[1]; s2 ^= k[2]; s3 ^= k[3]
        for r in range(1, self.nr):
            k = rkw[r]
            t0 = te0[s0 >> 24] ^ te1[(s1 >> 16) & 255] ^ te2[(s2 >> 8) & 255] ^ te3[s3 & 255] ^ k[0]
            t1 = te0[s1 >> 24] ^ te1[(s2 >> 16) & 255] ^ te2[(s3 >> 8) & 255] ^ te3[s0 & 255] ^ k[1]
            t2 = te0[s2 >> 24] ^ te1[(s3 >> 16) & 255] ^ te2[(s0 >> 8) & 255] ^ te3[s1 & 255] ^ k[2]
            t3 = te0[s3 >> 24] ^ te1[(s0 >> 16) & 255] ^ te2[(s1 >> 8) & 255] ^ te3[s2 & 255] ^ k[3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        k = rkw[self.nr]
        o0 = ((S[s0 >> 24] << 24) | (S[(s1 >> 16) & 255] << 16) | (S[(s2 >> 8) & 255] << 8) | S[s3 & 255]) ^ k[0]
        o1 = ((S[s1 >> 24] << 24) | (S[(s2 >> 16) & 255] << 16) | (S[(s3 >> 8) & 255] << 8) | S[s0 & 255]) ^ k[1]
        o2 = ((S[s2 >> 24] << 24) | (S[(s3 >> 16) & 255] << 16) | (S[(s0 >> 8) & 255] << 8) | S[s1 & 255]) ^ k[2]
        o3 = ((S[s3 >> 24] << 24) | (S[(s0 >> 16) & 255] << 16) | (S[(s1 >> 8) & 255] << 8) | S[s2 & 255]) ^ k[3]
        return o0, o1, o2, o3

    def _cbc_encrypt_blocks(self, blocks: np.ndarray, iv: bytes) -> np.ndarray:
        words = np.frombuffer(np.ascontiguousarray(blocks).tobytes(), dtype=">u4").reshape(-1, 4).tolist()
        p0, p1, p2, p3 = np.frombuffer(iv, dtype=">u4").tolist()
        out = []
        enc = self._enc_words
        for w0, w1, w2, w3 in words:
            p0, p1, p2, p3 = enc(w0 ^ p0, w1 ^ p1, w2 ^ p2, w3 ^ p3)
            out.append((p0, p1, p2, p3))
        return np.frombuffer(np.array(out, dtype=">u4").tobytes(), dtype=np.uint8).reshape(-1, 16)

    # --------------------------------------------------------------- API byte
    @staticmethod
    def _pad(data: bytes) -> tuple[np.ndarray, int]:
        n = len(data)
        padded = data + bytes((-n) % 16)                        # padding nol; panjang asli disimpan pemanggil
        return np.frombuffer(padded, dtype=np.uint8).reshape(-1, 16), n

    def encrypt(self, data: bytes, mode: str = "CBC", iv: bytes | None = None) -> bytes:
        """Kembalikan ciphertext sepanjang data yang dipad nol ke kelipatan 16."""
        blocks, _ = self._pad(data)
        mode = mode.upper()
        if mode == "ECB":
            return self.encrypt_blocks(blocks).tobytes()
        if mode == "CBC":
            iv = iv if iv is not None else bytes(16)
            return self._cbc_encrypt_blocks(blocks, iv).tobytes()
        raise ValueError("Mode harus ECB atau CBC.")

    def decrypt(self, cipher: bytes, orig_len: int, mode: str = "CBC", iv: bytes | None = None) -> bytes:
        blocks = np.frombuffer(cipher, dtype=np.uint8).reshape(-1, 16)
        mode = mode.upper()
        if mode == "ECB":
            plain = self.decrypt_blocks(blocks)
        elif mode == "CBC":
            iv = iv if iv is not None else bytes(16)
            prev = np.vstack([np.frombuffer(iv, dtype=np.uint8)[None, :], blocks[:-1]])
            plain = self.decrypt_blocks(blocks) ^ prev
        else:
            raise ValueError("Mode harus ECB atau CBC.")
        return plain.tobytes()[:orig_len]
