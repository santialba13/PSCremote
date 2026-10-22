"""Compara NXRE (realismo) del camino de GPU (gpu_emu modo 11: avgpool + red + re_ema +
re_apply) contra sr/net_re.py con los mismos pesos.

  make -C plugin/test all && python3 plugin/test/test_re_path.py weights/nxre.pkl imagen.png
"""
import os
import pickle
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sr"))
import net_re  # noqa: E402


def main():
    w, src = sys.argv[1], sys.argv[2]
    img = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB")))
    H, W = img.shape[:2]
    p = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(w, "rb")))
    ref = np.asarray(net_re.apply(p, jnp.asarray(img[None].astype(np.float32) / 255), 1.0))[0]
    with tempfile.TemporaryDirectory() as td:
        f, out, real = (os.path.join(td, n) for n in ("f.rgba", "mid.rgba", "real.rgba"))
        np.dstack([img, np.full((H, W), 255, np.uint8)]).tofile(f)
        subprocess.run([os.path.join(ROOT, "plugin", "test", "gpu_emu"), str(W), str(H), f, f, out, "11", real],
                       check=True, stdout=subprocess.DEVNULL)
        emu = np.fromfile(real, np.uint8).reshape(H, W, 4)[..., :3].astype(np.float32)
    d = np.abs(emu - np.round(np.clip(ref, 0, 1) * 255))
    print(f"{W}x{H}: max dif {d.max():.0f}/255, pixeles distintos {100 * (d > 0).any(-1).mean():.3f} %")
    print(f"cambio medio que aplica NXRE: {np.abs(emu - img).mean():.2f}/255")
    ok = d.max() <= 2
    print("OK" if ok else "DIFERENCIA DEMASIADO GRANDE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
