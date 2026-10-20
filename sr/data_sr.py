"""Datos para NXSR (realce / reescalado del fotograma final del juego).

Cada ejemplo es un par (entrada degradada, objetivo limpio) del mismo tamano.
Las degradaciones imitan lo que llega al framebuffer de un juego de PS4:
  * render a menor resolucion (escala 1.0-2.0) y reescalado bilineal o
    bicubico al tamano de salida (lo que hacen el juego o VideoOut);
  * desenfoque leve (TAA / post-proceso);
  * grano o dithering (ruido) y cuantizacion a 8 bits.
El objetivo es siempre la imagen limpia: el modelo aprende a recuperar
detalle SIN amplificar el ruido (lo contrario del granulado de GNMScale).

  python sr/data_sr.py --images DIR [DIR ...] --videos v.mp4 --n 16000 --out train_sr.npz
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))


def load_images(dirs, min_side=128):
    imgs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.bmp"))):
            im = np.asarray(Image.open(p).convert("RGB"))
            if min(im.shape[:2]) >= min_side:
                imgs.append(im)
    return imgs


def degrade(hr, rng):
    """hr uint8 (P, P, 3) -> entrada degradada uint8 del mismo tamano."""
    P = hr.shape[0]
    im = Image.fromarray(hr)
    s = rng.choice([1.0, 1.0, 1.25, 1.5, 1.5, 2.0]) * rng.uniform(0.95, 1.05)
    if s > 1.02:
        lo = max(8, int(round(P / s)))
        im = im.resize((lo, lo), Image.BOX)
        im = im.resize((P, P), Image.BILINEAR if rng.random() < 0.6 else Image.BICUBIC)
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    x = np.asarray(im).astype(np.float32)
    if rng.random() < 0.4:  # grano / dithering
        x = x + rng.normal(0, rng.uniform(0.5, 5.0), x.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def sample_patches(imgs, n, P, rng):
    out_hr = np.empty((n, P, P, 3), np.uint8)
    for i in range(n):
        im = imgs[rng.integers(len(imgs))]
        H, W = im.shape[:2]
        y, x = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
        p = im[y:y + P, x:x + P]
        if rng.random() < 0.5:
            p = p[:, ::-1]
        if rng.random() < 0.5:
            p = np.rot90(p)
        out_hr[i] = p
    return out_hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--synth", type=int, default=0, help="escenas sinteticas tipo juego (model/data.py)")
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    imgs = load_images(a.images, a.patch)
    if a.videos:
        import data as fgdata
        for v in a.videos:
            fr = fgdata.decode_video(v)
            imgs += [fr[i] for i in range(0, len(fr), 3)]
    if a.synth:
        import data as fgdata
        gen = fgdata.SyntheticGame(rng, 256, imgs[:50])
        imgs += [(gen.triplet()[1] * 255).astype(np.uint8) for _ in range(a.synth)]
    print("imagenes fuente:", len(imgs), flush=True)
    hr = sample_patches(imgs, a.n, a.patch, rng)
    lr = np.stack([degrade(h, rng) for h in hr])
    np.savez(a.out, hr=hr, lr=lr)
    print(a.out, hr.shape)


if __name__ == "__main__":
    main()
