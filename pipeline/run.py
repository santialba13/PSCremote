"""Demo en PC del pipeline completo de VCFG-NX sobre un video.

  python pipeline/run.py entrada.mp4 salida.mp4 --weights weights/nxflow.pkl \
      [--render-scale 0.75] [--post rcas|easu|none] [--sharpness 0.25] [--compare]

Hace lo mismo que el plugin en la consola, fotograma a fotograma:
  1. (opcional) baja la resolucion para simular un juego que renderiza a 900p/720p;
  2. block-matching + NXFlow -> fotograma intermedio entre cada par;
  3. FSR1 (EASU + RCAS) sobre reales e intermedios;
  4. escribe el video al doble de fps. --compare agrega a la izquierda el
     resultado del enfoque clasico tipo VCFG v2.14 (block-matching + mezcla).
"""
import argparse
import os
import pickle
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "upscale"))
import baseline_bm  # noqa: E402
import data  # noqa: E402
import fsr1  # noqa: E402
import net  # noqa: E402


def resize_area(img, h, w):
    from PIL import Image
    return np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
        (w, h), Image.BOX)).astype(np.float32) / 255.0


def pad32(x):
    H, W = x.shape[:2]
    return np.pad(x, ((0, (-H) % 32), (0, (-W) % 32), (0, 0)), mode="edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "nxflow.pkl"))
    ap.add_argument("--render-scale", type=float, default=1.0)
    ap.add_argument("--post", choices=["none", "rcas", "easu"], default="rcas")
    ap.add_argument("--sharpness", type=float, default=0.25)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0, help="fps de entrada (salida = 2x)")
    a = ap.parse_args()

    with open(a.weights, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    infer = jax.jit(lambda p, i0, i1, bm: net.interpolate(p, i0, i1, bm)[0])

    frames = data.decode_video(a.src).astype(np.float32) / 255.0
    if a.max_frames:
        frames = frames[:a.max_frames]
    Ho, Wo = frames.shape[1:3]
    Hr, Wr = int(Ho * a.render_scale) // 2 * 2, int(Wo * a.render_scale) // 2 * 2
    post = a.post if (Hr, Wr) != (Ho, Wo) or a.post != "easu" else "rcas"

    def finish(img):
        if post == "none":
            return img if img.shape[:2] == (Ho, Wo) else resize_area(img, Ho, Wo)
        if post == "easu" or img.shape[:2] != (Ho, Wo):
            img = fsr1.easu(img, Ho, Wo)
        return fsr1.rcas(img, a.sharpness)

    out_w = Wo * (2 if a.compare else 1)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{out_w}x{Ho}", "-r", str(a.fps * 2), "-i", "-",
                            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", a.dst],
                           stdin=subprocess.PIPE)

    def emit(img, ref=None):
        if a.compare:
            img = np.concatenate([ref, img], 1)
        enc.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

    rend = [f if (Hr, Wr) == (Ho, Wo) else resize_area(f, Hr, Wr) for f in frames]
    for i in range(len(rend)):
        real = finish(rend[i])
        if i > 0:
            i0, i1 = pad32(rend[i - 1]), pad32(rend[i])
            bm = baseline_bm.block_vectors(i0, i1)
            mid = np.asarray(infer(params, i0[None], i1[None], bm[None]))[0][:Hr, :Wr]
            ref = None
            if a.compare:
                ref = finish(np.clip(baseline_bm.interpolate(i0, i1), 0, 1)[:Hr, :Wr])
            emit(finish(np.clip(mid, 0, 1)), ref)
        emit(real, real if a.compare else None)
        print(f"\r{i + 1}/{len(rend)}", end="", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"\n{a.dst}: {2 * len(rend) - 1} fotogramas a {a.fps * 2:g} fps")


if __name__ == "__main__":
    main()
