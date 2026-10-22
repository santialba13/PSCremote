"""Compara el camino de GPU (DCB del plugin ejecutado por gpu_emu con los kernels
.cl) contra el modelo de referencia en NumPy, sobre fotogramas reales.

  make -C plugin/test all
  python3 plugin/test/test_gpu_path.py video.mp4 [--frame 10]

Compara la generacion NXFG-3 (model/net3.py, referencia NumPy: reference_runtime.interpolate3),
la rapida (modo 6), RCAS y FSR+.
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "upscale"))
import data  # noqa: E402
import fsr1  # noqa: E402
import reference_runtime as rr  # noqa: E402


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("weights", nargs="?", help="(sin uso desde NXFG-3)")
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--emu", default=os.path.join(ROOT, "plugin", "test", "gpu_emu"))
    a = ap.parse_args()

    fr = data.decode_video(a.video)
    f0, gt, f1 = fr[a.frame], fr[a.frame + 1], fr[a.frame + 2]
    H, W = f0.shape[:2]
    params = None

    with tempfile.TemporaryDirectory() as td:
        paths = [os.path.join(td, n) for n in ("f0", "f1", "mid", "mid_rcas", "real_rcas", "mid_fast", "x", "plus")]
        for img, pth in zip((f0, f1), paths):
            np.dstack([img, np.full((H, W), 255, np.uint8)]).tofile(pth)
        subprocess.run([a.emu, str(W), str(H), paths[0], paths[1], paths[2]], check=True)
        subprocess.run([a.emu, str(W), str(H), paths[0], paths[1], paths[3], "1", paths[4]],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([a.emu, str(W), str(H), paths[0], paths[1], paths[5], "6"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([a.emu, str(W), str(H), paths[0], paths[1], paths[6], "7", paths[7]], check=True,
                       stdout=subprocess.DEVNULL)
        load = lambda p: np.fromfile(p, np.uint8).reshape(H, W, 4)[..., :3].astype(np.float32) / 255
        emu, emu_rcas, real_rcas, emu_fast = load(paths[2]), load(paths[3]), load(paths[4]), load(paths[5])
        emu_plus = load(paths[7])

    i0 = rr.np.pad(f0.astype(np.float32) / 255, ((0, (-H) % 32), (0, (-W) % 32), (0, 0)), mode="edge")
    i1 = rr.np.pad(f1.astype(np.float32) / 255, ((0, (-H) % 32), (0, (-W) % 32), (0, 0)), mode="edge")
    ref = np.clip(rr.interpolate3(i0, i1)[:H, :W], 0, 1)
    ref_u8 = np.round(ref * 255) / 255
    ref_fast = np.round(np.clip(rr.interpolate2(params, i0, i1, fast=True)[:H, :W], 0, 1) * 255) / 255
    g = gt.astype(np.float32) / 255

    print(f"{W}x{H}, fotograma {a.frame}")
    print(f"  GPU emulada vs referencia NumPy : {psnr(emu, ref_u8):6.2f} dB  (max dif {np.abs(emu - ref_u8).max() * 255:.0f}/255)")
    print(f"  referencia vs fotograma real    : {psnr(ref, g):6.2f} dB")
    print(f"  GPU emulada vs fotograma real   : {psnr(emu, g):6.2f} dB")
    print(f"  rapida: GPU vs referencia      : {psnr(emu_fast, ref_fast):6.2f} dB  (vs real {psnr(emu_fast, g):.2f} dB)")
    print(f"  mezcla 50/50 vs fotograma real  : {psnr(0.5 * (f0.astype(np.float32) + f1) / 255, g):6.2f} dB")
    print(f"  RCAS GPU vs fsr1.rcas (generado: {psnr(emu_rcas, fsr1.rcas(emu, 0.25)):6.2f} dB")
    print(f"  RCAS GPU vs fsr1.rcas (real)    : {psnr(real_rcas, fsr1.rcas(f1.astype(np.float32) / 255, 0.25)):6.2f} dB")
    ref_plus = fsr1.rcas(f1.astype(np.float32) / 255, 0.25, detail=0.6)
    print(f"  FSR+ GPU vs fsr1.rcas(detalle)  : {psnr(emu_plus, ref_plus):6.2f} dB  (cambio vs original "
          f"{psnr(ref_plus, f1.astype(np.float32) / 255):.1f} dB)")
    ok = psnr(emu_plus, ref_plus) > 40 and psnr(emu, ref_u8) > 40 and psnr(emu_fast, ref_fast) > 40 and psnr(emu_rcas, fsr1.rcas(emu, 0.25)) > 40
    print("OK" if ok else "DIFERENCIA DEMASIADO GRANDE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
