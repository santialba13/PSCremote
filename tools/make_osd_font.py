"""Genera plugin/include/nx_osd_font.h: la fuente del menu dentro del juego.

DejaVu Sans Mono Bold (licencia libre de Bitstream Vera / DejaVu) renderizada con
suavizado (alfa de 8 bits) en celdas fijas, para Latin-1 (32..255). El kernel
osd_draw la lee como atlas: [glifo][fila][columna].

  python3 tools/make_osd_font.py [ruta_a_DejaVuSansMono-Bold.ttf]
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
OUT = os.path.join(ROOT, "plugin", "include", "nx_osd_font.h")
SIZE, CW, CH = 25, 15, 30
FIRST, LAST = 32, 255


def main():
    f = ImageFont.truetype(FONT, SIZE)
    ascent, descent = f.getmetrics()
    top = (CH - (ascent + descent)) // 2
    data = bytearray()
    for code in range(FIRST, LAST + 1):
        im = Image.new("L", (CW, CH), 0)
        if not (0x7F <= code < 0xA0):
            ImageDraw.Draw(im).text((0, top), chr(code), font=f, fill=255)
        data += im.tobytes()
    lines = ["// Generado por tools/make_osd_font.py: no editar a mano.",
             "// DejaVu Sans Mono Bold (Bitstream Vera / DejaVu, licencia libre), suavizada.",
             "#pragma once", "#include <stdint.h>",
             f"#define NX_OSD_CW {CW}", f"#define NX_OSD_CH {CH}",
             f"#define NX_OSD_FIRST {FIRST}", f"#define NX_OSD_GLYPHS {LAST - FIRST + 1}",
             "// el atlas solo en el archivo que define NX_OSD_FONT_DATA (son ~100 KiB)",
             "#ifdef NX_OSD_FONT_DATA",
             f"static const uint8_t nx_osd_font[{len(data)}] __attribute__((aligned(256))) = {{"]
    for i in range(0, len(data), 40):
        lines.append("    " + ",".join(str(b) for b in data[i:i + 40]) + ",")
    lines.append("};")
    lines.append("#endif")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(OUT, f"{len(data) // 1024} KiB")


if __name__ == "__main__":
    main()
