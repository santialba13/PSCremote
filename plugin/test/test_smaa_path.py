"""Compara el SMAA del camino de GPU (gpu_emu, modo 2) con upscale/smaa.py
sobre la escena de prueba con bordes duros, y ambos contra SSAA 8x8.

  make -C plugin/test all && python3 plugin/test/test_smaa_path.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "upscale"))
import aa_testscene  # noqa: E402
import fsr1  # noqa: E402
import smaa  # noqa: E402


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((np.clip(a, 0, 1) - b) ** 2), 1e-12))


def main():
    W, H = 256, 160
    gt = aa_testscene.render(W, H, ss=8)
    al8 = np.round(aa_testscene.render(W, H, ss=1) * 255).astype(np.uint8)
    al = al8.astype(np.float64) / 255
    emu = os.path.join(ROOT, "plugin", "test", "gpu_emu")
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "f.rgba")
        np.dstack([al8, np.full((H, W), 255, np.uint8)]).tofile(f)
        mid, real = os.path.join(td, "mid"), os.path.join(td, "real")
        subprocess.run([emu, str(W), str(H), f, f, mid, "2", real], check=True, stdout=subprocess.DEVNULL)
        gpu = np.fromfile(real, np.uint8).reshape(H, W, 4)[..., :3].astype(np.float64) / 255
    ref = smaa.smaa(al)
    ref8 = np.floor(np.clip(ref, 0, 1) * 255 + 0.5) / 255
    diff = np.abs(gpu - ref8).max() * 255
    print(f"SMAA GPU emulado vs referencia Python : {psnr(gpu, ref8):6.2f} dB (max dif {diff:.0f}/255)")
    print(f"vs SSAA 8x8 -> sin AA {psnr(al, gt):.2f} dB | RCAS (FSR1) {psnr(fsr1.rcas(al.astype(np.float32), 0.25), gt):.2f} dB"
          f" | SMAA referencia {psnr(ref, gt):.2f} dB | SMAA GPU {psnr(gpu, gt):.2f} dB")
    ok = psnr(gpu, ref8) > 40
    print("OK" if ok else "DIFERENCIA DEMASIADO GRANDE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
