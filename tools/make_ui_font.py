"""Genera installer/source/ui_font_data.h: la fuente Inter (SIL OFL 1.1)
pre-renderizada con suavizado (alfa de 8 bits) en los tamanos que usa la app,
y los iconos de los botones del mando (X, O, cuadrado, triangulo) en ARGB.

  python3 tools/make_ui_font.py [carpeta_de_Inter]

Cada glifo guarda su caja ajustada (sin ceros alrededor): ~0,5 MB en total para
Latin-1. Renderizar aca, con FreeType y hinting, se ve mejor y es mas rapido que
rasterizar en la consola.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/opentype/inter"
OUT = os.path.join(ROOT, "installer", "source", "ui_font_data.h")

# (nombre, archivo, tamano en px)
FONTS = [
    ("FONT_LOGO", "Inter-ExtraBold.otf", 50),
    ("FONT_H1", "Inter-Bold.otf", 36),
    ("FONT_H2", "Inter-SemiBold.otf", 27),
    ("FONT_BODY", "Inter-Medium.otf", 25),
    ("FONT_CARD", "Inter-SemiBold.otf", 22),
    ("FONT_SMALL", "Inter-Medium.otf", 20),
    ("FONT_TINY", "Inter-Bold.otf", 15),
]
FIRST, LAST = 32, 255


def render_font(path, size):
    f = ImageFont.truetype(path, size)
    ascent, descent = f.getmetrics()
    glyphs, data = [], bytearray()
    for code in range(FIRST, LAST + 1):
        ch = chr(code)
        if code < 0xA0 and code > 0x7E:  # controles C1: sin glifo
            glyphs.append((0, 0, 0, 0, 0, len(data)))
            continue
        adv = int(round(f.getlength(ch)))
        box = f.getbbox(ch)  # relativo al origen (arriba-izquierda de la linea)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            glyphs.append((adv, 0, 0, 0, 0, len(data)))
            continue
        x0, y0, x1, y1 = box
        im = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(im).text((-x0, -y0), ch, font=f, fill=255)
        glyphs.append((adv, x0, y0, x1 - x0, y1 - y0, len(data)))
        data += im.tobytes()
    return ascent, descent, glyphs, bytes(data)


def button_sprite(kind, size=40):
    """Icono de boton estilo PlayStation: disco oscuro con el simbolo de color."""
    ss = 4
    S = size * ss
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([ss, ss, S - ss - 1, S - ss - 1], fill=(32, 38, 56, 255), outline=(255, 255, 255, 40), width=ss)
    col = {"cross": (110, 160, 255), "circle": (255, 98, 108), "square": (236, 140, 206), "triangle": (70, 214, 162)}[kind]
    w = int(S * 0.085)
    c = S / 2
    r = S * 0.24
    if kind == "cross":
        d.line([(c - r, c - r), (c + r, c + r)], fill=col, width=w)
        d.line([(c - r, c + r), (c + r, c - r)], fill=col, width=w)
    elif kind == "circle":
        d.ellipse([c - r, c - r, c + r, c + r], outline=col, width=w)
    elif kind == "square":
        d.rectangle([c - r * 0.9, c - r * 0.9, c + r * 0.9, c + r * 0.9], outline=col, width=w)
    else:
        h = r * 1.05
        d.polygon([(c, c - h), (c + h * 1.1, c + h * 0.75), (c - h * 1.1, c + h * 0.75)], outline=col, width=w)
    im = im.resize((size, size), Image.LANCZOS)
    raw = im.tobytes()
    return [(raw[i + 3] << 24) | (raw[i] << 16) | (raw[i + 1] << 8) | raw[i + 2] for i in range(0, len(raw), 4)]


def carray(name, ctype, values, per_line=24):
    out = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        out.append("    " + ",".join(str(v) for v in values[i:i + per_line]) + ",")
    out.append("};")
    return out


def main():
    lines = ["// Generado por tools/make_ui_font.py: no editar a mano.",
             "// Fuente: Inter (c) 2016-2023 The Inter Project Authors, SIL Open Font License 1.1",
             "// (third_party/inter/OFL.txt). Glifos pre-renderizados = obra derivada bajo la misma licencia.",
             "#pragma once", "#include <stdint.h>", "",
             "typedef struct { int16_t adv, x, y; uint8_t w, h; uint32_t off; } ui_glyph_t;",
             "typedef struct { int size, ascent, descent; const ui_glyph_t* g; const uint8_t* bits; } ui_font_t;",
             f"#define UI_FONT_FIRST {FIRST}", f"#define UI_FONT_LAST {LAST}", ""]
    total = 0
    tables = []
    for idx, (name, file, size) in enumerate(FONTS):
        asc, desc, glyphs, data = render_font(os.path.join(FONT_DIR, file), size)
        total += len(data)
        lname = name.lower()
        lines.append(f"static const ui_glyph_t {lname}_g[{len(glyphs)}] = {{")
        for g in glyphs:
            lines.append("    {%d,%d,%d,%d,%d,%d}," % g)
        lines.append("};")
        lines += carray(f"{lname}_bits", "uint8_t", list(data) or [0], 40)
        tables.append(f"    {{ {size}, {asc}, {desc}, {lname}_g, {lname}_bits }},  // {name} {file}")
        lines.append(f"#define {name} {idx}")
        lines.append("")
    lines.append(f"#define UI_NUM_FONTS {len(FONTS)}")
    lines.append("static const ui_font_t ui_fonts[UI_NUM_FONTS] = {")
    lines += tables
    lines.append("};")
    lines.append("")
    lines.append("#define UI_BTN_SIZE 40")
    for k in ("cross", "circle", "square", "triangle"):
        lines += carray(f"ui_btn_{k}", "uint32_t", button_sprite(k), 12)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(OUT, f"{total / 1024:.0f} KiB de glifos")


if __name__ == "__main__":
    main()
