"""Datos de entrenamiento para VCFG-NX.

Dos fuentes:
  * Escenas sinteticas "tipo juego": fondo con paneo de camara (parallax),
    sprites que se mueven y se tapan entre si, HUD estatico y fundidos.
    El fotograma intermedio se renderiza de forma exacta en t=0.5, asi que
    el ground truth es perfecto y cubre los casos que mas fallan en un
    frame generator de consola (HUD, oclusiones, cortes de escena).
  * Tripletes reales (f[i], f[i+1], f[i+2]) de videos decodificados con ffmpeg.

Todas las imagenes son float32 RGB en [0,1], forma (H, W, 3).
"""
import os
import subprocess
import numpy as np


def decode_video(path, max_side=None):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    vf = []
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(w * s) // 2 * 2, int(h * s) // 2 * 2
        vf = ["-vf", f"scale={w}:{h}:flags=area"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def video_triplets(frames, crop, rng, n, min_motion=0.004):
    """Muestrea n tripletes recortados; descarta los casi estaticos."""
    T, H, W, _ = frames.shape
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        t = rng.integers(0, T - 2)
        y = rng.integers(0, H - crop + 1)
        x = rng.integers(0, W - crop + 1)
        tri = frames[t:t + 3, y:y + crop, x:x + crop].astype(np.float32) / 255.0
        if np.abs(tri[0] - tri[2]).mean() < min_motion:
            continue
        if rng.random() < 0.5:
            tri = tri[::-1]  # inversion temporal
        if rng.random() < 0.5:
            tri = tri[:, :, ::-1]
        out.append(np.ascontiguousarray(tri))
    return out


# ---------------------------------------------------------------- sintetico

def _smooth_noise(rng, h, w, octaves=5):
    img = np.zeros((h, w, 3), np.float32)
    amp = 1.0
    for o in range(octaves):
        gh, gw = max(2, h >> (octaves - o)), max(2, w >> (octaves - o))
        g = rng.random((gh, gw, 3)).astype(np.float32)
        yy = np.linspace(0, gh - 1, h)
        xx = np.linspace(0, gw - 1, w)
        y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
        y1 = np.minimum(y0 + 1, gh - 1); x1 = np.minimum(x0 + 1, gw - 1)
        fy = (yy - y0)[:, None, None]; fx = (xx - x0)[None, :, None]
        up = (g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx) * (1 - fy) + \
             (g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx) * fy
        img += amp * up
        amp *= 0.55
    img -= img.min(); img /= img.max() + 1e-6
    # bordes duros para que parezca geometria de juego, no solo ruido
    if rng.random() < 0.7:
        k = rng.integers(3, 8)
        img = np.floor(img * k) / k * 0.6 + img * 0.4
    tint = rng.random(3).astype(np.float32) * 0.6 + 0.4
    return np.clip(img * tint, 0, 1)


def _sample(tex, y, x):
    """Muestreo bilineal con wrap; y, x son arrays (H, W)."""
    th, tw, _ = tex.shape
    y = np.mod(y, th - 1); x = np.mod(x, tw - 1)
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    fy = (y - y0)[..., None]; fx = (x - x0)[..., None]
    return (tex[y0, x0] * (1 - fx) + tex[y0, x0 + 1] * fx) * (1 - fy) + \
           (tex[y0 + 1, x0] * (1 - fx) + tex[y0 + 1, x0 + 1] * fx) * fy


class SyntheticGame:
    def __init__(self, rng, size, textures=None):
        self.rng = rng
        self.size = size
        self.textures = textures or []

    def _texture(self, h, w):
        if self.textures and self.rng.random() < 0.6:
            t = self.textures[self.rng.integers(len(self.textures))]
            th, tw, _ = t.shape
            if th > h and tw > w:
                y = self.rng.integers(0, th - h); x = self.rng.integers(0, tw - w)
                return t[y:y + h, x:x + w].astype(np.float32) / 255.0
        return _smooth_noise(self.rng, h, w)

    def triplet(self):
        r, S = self.rng, self.size
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        bg = self._texture(S * 2, S * 2)
        # camara: traslacion + zoom leve + rotacion leve (giro de camara en 3a persona)
        v = r.normal(0, 1, 2) * r.choice([2, 6, 14, 28])
        zoom = 1.0 + r.normal(0, 0.015)
        rot = r.normal(0, 0.01)
        objs = []
        for _ in range(r.integers(0, 5)):
            oh, ow = r.integers(S // 10, S // 2, 2)
            objs.append(dict(
                tex=self._texture(oh, ow), h=oh, w=ow,
                p=r.random(2) * S, v=r.normal(0, 1, 2) * r.choice([3, 10, 24, 40]),
                round=r.random() < 0.5))
        hud = []
        for _ in range(r.integers(0, 4)):
            hh, hw = r.integers(4, S // 6), r.integers(S // 8, S // 2)
            hud.append((r.integers(0, S - hh), r.integers(0, S - hw), hh, hw,
                        r.random(3) * 0.8 + 0.2, r.random() * 0.5 + 0.5))
        fade = r.normal(0, 0.06) if r.random() < 0.15 else 0.0

        def render(t):
            a = rot * t; z = zoom ** t
            c, s = np.cos(a), np.sin(a)
            cy, cx = yy - S / 2, xx - S / 2
            sy = (c * cy - s * cx) / z + S / 2 + v[0] * t + S / 2
            sx = (s * cy + c * cx) / z + S / 2 + v[1] * t + S / 2
            img = _sample(bg, sy, sx)
            for o in objs:
                py, px = o["p"] + o["v"] * t
                ly, lx = yy - py, xx - px
                m = (ly >= 0) & (ly < o["h"] - 1) & (lx >= 0) & (lx < o["w"] - 1)
                if o["round"]:
                    m &= ((ly / o["h"] - .5) ** 2 + (lx / o["w"] - .5) ** 2) < .25
                if m.any():
                    img[m] = _sample(o["tex"], ly[m], lx[m])
            img = img * (1 + fade * t)
            for (y0, x0, hh, hw, col, al) in hud:
                img[y0:y0 + hh, x0:x0 + hw] = img[y0:y0 + hh, x0:x0 + hw] * (1 - al) + col * al
            return np.clip(img, 0, 1).astype(np.float32)

        return np.stack([render(-1.0), render(0.0), render(1.0)])


def build_dataset(video_paths, n_video, n_synth, crop, seed=0, max_side=720):
    rng = np.random.default_rng(seed)
    clips = [decode_video(p, max_side) for p in video_paths]
    per = n_video // len(clips) if n_video and clips else 0
    # se escribe directo en float16 para no duplicar memoria con miles de recortes
    arr = np.empty((per * len(clips) + n_synth, 3, crop, crop, 3), np.float16)
    n = 0
    for c in clips:
        for t in video_triplets(c, crop, rng, per):
            arr[n] = t
            n += 1
    textures = [c[i] for c in clips for i in range(0, len(c), 25)]
    gen = SyntheticGame(rng, crop, textures)
    for _ in range(n_synth):
        arr[n] = gen.triplet()
        n += 1
    arr = arr[:n]
    rng.shuffle(arr)
    return arr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--n-video", type=int, default=3000)
    ap.add_argument("--n-synth", type=int, default=3000)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arr = build_dataset(a.videos, a.n_video, a.n_synth, a.crop, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, arr)
    print(a.out, arr.shape)
