"""Precalcula el movimiento de NXFG-2 (vectores + confianza, malla 1/16) de cada triplete.

  python model/precompute_bm2.py train.npy train_bm2.npy
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bm2  # noqa: E402

_DATA = None


def _init(path):
    global _DATA
    _DATA = np.load(path, mmap_mode="r")


def _one(i):
    t = _DATA[i].astype(np.float32)
    return bm2.features(t[0], t[2])[4]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = np.load(src, mmap_mode="r").shape[0]
    with Pool(os.cpu_count(), initializer=_init, initargs=(src,)) as pool:
        vecs = pool.map(_one, range(n), chunksize=16)
    np.save(dst, np.stack(vecs).astype(np.float32))
    print(dst, len(vecs))


if __name__ == "__main__":
    main()
