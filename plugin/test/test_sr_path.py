"""Compara NXSR del camino de GPU (gpu_emu modo 3: kernel wg_nxsr con LDS y
barreras emuladas) contra sr/net_sr.py con los mismos pesos.

  make -C plugin/test all && python3 plugin/test/test_sr_path.py weights/nxsr.pkl imagen.png
"""
import os
import pickle
import subprocess
import sys
import tempfile

import jax.numpy as jnp
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sr"))
import net_sr  # noqa: E402


def main():
    w, src = sys.argv[1], sys.argv[2]
    img = np.asarray(Image.open(src).convert("RGB"))
    H, W = img.shape[0] // 16 * 16, img.shape[1] // 16 * 16
    img = img[:H, :W]
    p = {k: {kk: jnp.asarray(vv) for kk, vv in v.items()} for k, v in pickle.load(open(w, "rb")).items()}
    ref = np.asarray(net_sr.apply(p, jnp.asarray(img[None].astype(np.float32) / 255), 1.0, True, 0.0))[0]
    ref8 = np.floor(np.clip(ref, 0, 1) * 255 + 0.5)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "f")
        np.dstack([img, np.full((H, W), 255, np.uint8)]).tofile(f)
        out, real = os.path.join(td, "o"), os.path.join(td, "r")
        subprocess.run([os.path.join(ROOT, "plugin", "test", "gpu_emu"), str(W), str(H), f, f, out, "3", real],
                       check=True, stdout=subprocess.DEVNULL)
        gpu = np.fromfile(real, np.uint8).reshape(H, W, 4)[..., :3].astype(np.float64)
    d = np.abs(gpu - ref8)
    print(f"{W}x{H}: max dif {d.max():.0f}/255, pixeles distintos {100 * (d > 0).any(-1).mean():.3f} %")
    print(f"cambio medio que aplica NXSR: {np.abs(ref8 - img).mean():.2f}/255")
    ok = d.max() <= 1
    print("OK" if ok else "DIFERENCIA DEMASIADO GRANDE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
