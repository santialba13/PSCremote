"""Genera plugin/include/nx_font.h: fuente bitmap 8x16 (Latin-1, codigos 32-255)
rasterizada de DejaVu Sans Mono Bold (licencia Bitstream Vera / DejaVu, libre).

  python3 tools/make_font.py /usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf plugin/include/nx_font.h
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
font = ImageFont.truetype(src, 14)
rows = []
for code in range(32, 256):
    ch = bytes([code]).decode("latin-1")
    img = Image.new("L", (8, 16), 0)
    d = ImageDraw.Draw(img)
    d.text((0, 0), ch, font=font, fill=255)
    a = np.asarray(img) > 100
    for r in range(16):
        rows.append(int("".join("1" if b else "0" for b in a[r]), 2))
out = ["// Generado por tools/make_font.py (DejaVu Sans Mono Bold, 8x16, Latin-1 32..255).",
       "#pragma once", "#include <stdint.h>", "#define NX_FONT_FIRST 32", "#define NX_FONT_COUNT 224",
       "// 16 bytes por glifo; bit 7 = columna izquierda",
       "static const uint8_t nx_font8x16[224 * 16] = {"]
for i in range(0, len(rows), 16):
    out.append("    " + ", ".join(f"0x{v:02x}" for v in rows[i:i + 16]) + ",")
out.append("};")
open(dst, "w").write("\n".join(out) + "\n")
# vista previa
prev = np.zeros((16 * 7, 8 * 32), np.uint8)
for i in range(224):
    g = np.array([[(rows[i * 16 + r] >> (7 - c)) & 1 for c in range(8)] for r in range(16)]) * 255
    prev[(i // 32) * 16:(i // 32) * 16 + 16, (i % 32) * 8:(i % 32) * 8 + 8] = g
Image.fromarray(prev).resize((8 * 32 * 3, 16 * 7 * 3), Image.NEAREST).save(dst.replace(".h", "_preview.png"))
print(dst)
