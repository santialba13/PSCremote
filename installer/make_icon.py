"""Genera sce_sys/icon0.png (512x512) para el PKG del instalador.

Estilo "Aurora" (el mismo de la app): grafito casi negro, dos luces difusas
turquesa y violeta, una placa de vidrio con borde que brilla arriba y la marca
"vcfg nx" en Inter Display liviana.
"""
import math
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W = 512
SS = 2  # supermuestreo para bordes suaves
S = W * SS


def font(size, names):
    for f in names:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


INTER = "/usr/share/fonts/opentype/inter/"
light = font(112 * SS, [INTER + "InterDisplay-Light.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
small = font(34 * SS, [INTER + "Inter-SemiBold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"])

# fondo: grafito + auroras
img = Image.new("RGB", (S, S), (8, 9, 13))
px = img.load()
blobs = [(0.18, 0.05, 0.75, 0.55, (20, 184, 166), 0.55), (0.92, 0.70, 0.70, 0.65, (124, 92, 246), 0.55),
         (0.50, 1.10, 0.80, 0.40, (236, 72, 153), 0.22)]
for y in range(S):
    for x in range(S):
        r, g, b = 8.0, 9.0, 13.0
        for bx, by, rx, ry, col, k in blobs:
            dx, dy = (x / S - bx) / rx, (y / S - by) / ry
            q = 1 - (dx * dx + dy * dy)
            if q > 0:
                q = q * q * k
                r += col[0] * q
                g += col[1] * q
                b += col[2] * q
        px[x, y] = (min(255, int(r)), min(255, int(g)), min(255, int(b)))

# placa de vidrio: el fondo desenfocado y aclarado dentro de un rectangulo redondeado
m = 70 * SS
box = [m, 96 * SS, S - m, S - 96 * SS]
frost = img.filter(ImageFilter.GaussianBlur(28 * SS)).point(lambda v: min(255, int(v * 1.25 + 16)))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle(box, 54 * SS, fill=255)
img.paste(frost, (0, 0), mask)
over = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(over)
# borde: brillante arriba, casi invisible abajo
edge = Image.new("L", (S, S), 0)
ImageDraw.Draw(edge).rounded_rectangle(box, 54 * SS, outline=255, width=3 * SS)
grad = Image.new("L", (S, S), 0)
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = (y - box[1]) / (box[3] - box[1])
    gd.line([(0, y), (S, y)], fill=max(0, min(255, int(200 - 170 * t))))
edge = Image.composite(grad, Image.new("L", (S, S), 0), edge)
over.paste((255, 255, 255, 255), (0, 0), edge)
img = Image.alpha_composite(img.convert("RGBA"), over)

# marca: "vcfg" blanco + "nx" turquesa, y una gota de luz
d = ImageDraw.Draw(img)
cx, cy = S // 2, S // 2 - 6 * SS
w1 = d.textlength("vcfg", font=light)
w2 = d.textlength("nx", font=light)
x0 = cx - (w1 + w2) / 2
d.text((x0, cy), "vcfg", font=light, fill=(244, 246, 250), anchor="lm")
d.text((x0 + w1, cy), "nx", font=light, fill=(63, 224, 207), anchor="lm")
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(glow).ellipse([cx - 16 * SS, box[1] + 40 * SS, cx + 16 * SS, box[1] + 72 * SS], fill=(124, 240, 226, 255))
glow = glow.filter(ImageFilter.GaussianBlur(7 * SS))
img = Image.alpha_composite(img, glow)
d = ImageDraw.Draw(img)
d.text((cx, box[3] - 54 * SS), "FRAME GEN  ·  IA", font=small, fill=(167, 175, 190), anchor="mm")

img = img.convert("RGB").resize((W, W), Image.LANCZOS)
img.save(sys.argv[1] if len(sys.argv) > 1 else "icon0.png")
