"""Ejecuta cada modo del pipeline con el CODIGO MAQUINA GCN real (tools/gcn_isa_emu.py,
emulador de la GPU de PS4 a nivel de instrucciones) y lo compara con gpu_emu (los
mismos kernels recompilados para x86).

  make -C plugin/test gpu_emu && python3 plugin/test/test_isa_path.py video.mp4 [--frame 10]

Sirve para detectar lo que gpu_emu no ve: un error del compilador para gfx701, un
descriptor (T#/S#/V#) o un kernarg mal armado, un acceso fuera de la memoria
reservada (en la consola: la GPU se cuelga o el juego se cierra), una instruccion
que la PS4 no tiene (FLAT), un bucle que no termina.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model"))
import data  # noqa: E402

MODES = [
    (0, "generacion IA (NXFG-2)", True, False),
    (6, "generacion rapida (sin red)", True, False),
    (1, "IA + nitidez RCAS", True, True),
    (7, "nitidez FSR+ (RCAS + detalle)", False, True),
    (2, "IA + SMAA", True, True),
    (3, "NXSR Rapido", False, True),
    (5, "NXSR Calidad", False, True),
    (4, "IA sobre reales procesados con NXSR", True, True),
    (8, "NXSR-L (ligero)", False, True),
    (9, "IA + NXSR-L + SMAA (reales procesados)", True, True),
]


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(np.mean((a - b) ** 2), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--crop", default="96x160+300+500", help="alto x ancho + y + x")
    ap.add_argument("--emu", default=os.path.join(ROOT, "plugin", "test", "gpu_emu"))
    a = ap.parse_args()
    hw, yx = a.crop.split("+", 1)
    H, W = (int(v) for v in hw.split("x"))
    y0, x0 = (int(v) for v in yx.split("+"))
    fr = data.decode_video(a.video)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        p = lambda n: os.path.join(td, n)
        for k, n in ((a.frame, "f0"), (a.frame + 2, "f1")):
            c = fr[k][y0:y0 + H, x0:x0 + W]
            np.dstack([c, np.full((H, W), 255, np.uint8)]).tofile(p(n + ".rgba"))
        load = lambda n: np.fromfile(p(n), np.uint8).reshape(H, W, 4)[..., :3].astype(np.float64)
        print("%dx%d, fotograma %d" % (W, H, a.frame))
        for mode, name, has_mid, has_real in MODES:
            env = dict(os.environ, NX_DUMP=p("job.bin"))
            subprocess.run([a.emu, str(W), str(H), p("f0.rgba"), p("f1.rgba"), p("mid_h.rgba"), str(mode),
                            p("real_h.rgba")], check=True, env=env, stdout=subprocess.DEVNULL)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gcn_isa_emu.py"), p("job.bin"),
                                "--out-mid", p("mid_i.rgba"), "--out-real", p("real_i.rgba")],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("  %-36s FALLO en el emulador de instrucciones:\n%s" % (name, r.stderr[-1500:]))
                ok = False
                continue
            stats = [l for l in r.stdout.splitlines() if l.startswith("dispatches")]
            res = []
            for has, h, i in ((has_mid, "mid_h.rgba", "mid_i.rgba"), (has_real, "real_h.rgba", "real_i.rgba")):
                if not has:
                    continue
                x, y = load(h), load(i)
                d = np.abs(x - y)
                res.append("%s %.1f dB (max %d)" % ("intermedio" if h.startswith("mid") else "real", psnr(x, y), d.max()))
                if psnr(x, y) < 50 or d.max() > 4:
                    ok = False
            print("  %-36s %s | %s" % (name, ", ".join(res), stats[0] if stats else ""))
    print("OK: el codigo maquina GCN da lo mismo que gpu_emu" if ok else "DIFERENCIAS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
