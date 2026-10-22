"""Pares de entrenamiento de NXRE desde un video lado a lado (izquierda: GTA V tal
cual; derecha: el mismo fotograma convertido a fotorrealista por un modelo de
investigacion, REGEN entrenado con resultados de EPE hacia Cityscapes).

  python sr/make_re_pairs.py test_rtx4090_720p.mp4 re_pairs.npz

La mitad derecha esta ampliada x1.236 y va 2 fotogramas atrasada: se alinea cada
par con ECC sobre los bordes (transformacion afin), se descartan los que no
alinean bien (cortes, movimiento rapido) y se recorta la zona comun. Salida a
mitad de resolucion (131 x 250).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

G = np.array([[1.236, 0, -68.0], [0, 1.240, -53.0]], np.float32)
X0, X1, Y0, Y1 = 68, 568, 56, 318
SPLIT = 626
FR = None


def edges(im):
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (m.mean() + 1e-6)


def work(i):
    L = np.ascontiguousarray(FR[i][:, :SPLIT])
    eL = edges(L)
    best = None
    for lag in (1, 2, 3):
        if i + lag >= len(FR):
            continue
        R = np.ascontiguousarray(FR[i + lag][:, SPLIT + 1:])
        W = G.copy()
        try:
            cc, W = cv2.findTransformECC(eL, edges(R), W, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error:
            continue
        if np.abs(W - G).max() > 8 or abs(W[0, 0] - 1.236) > 0.03 or abs(W[1, 1] - 1.24) > 0.03:
            continue
        if best is None or cc > best[0]:
            best = (cc, W, R)
    if best is None or best[0] < 0.6:
        return None
    cc, W, R = best
    Rw = cv2.warpAffine(R, W, (SPLIT, L.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    half = lambda im: cv2.resize(im[Y0:Y1, X0:X1], ((X1 - X0) // 2, (Y1 - Y0) // 2), interpolation=cv2.INTER_AREA)
    return half(L), half(Rw)


def main():
    global FR
    cap = cv2.VideoCapture(sys.argv[1])
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[..., ::-1])
    FR = np.stack(fr)
    with ProcessPoolExecutor(4) as ex:
        out = [o for o in ex.map(work, range(len(FR) - 3), chunksize=16) if o is not None]
    print(f"pares {len(out)} de {len(FR) - 3}")
    np.savez(sys.argv[2], src=np.stack([o[0] for o in out]), dst=np.stack([o[1] for o in out]))


if __name__ == "__main__":
    main()
