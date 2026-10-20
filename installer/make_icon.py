"""Genera sce_sys/icon0.png (512x512) para el PKG del instalador."""
import sys

from PIL import Image, ImageDraw, ImageFont

W = 512
img = Image.new("RGB", (W, W), (14, 18, 30))
d = ImageDraw.Draw(img)
for i in range(W):  # degradado
    c = int(40 + 60 * i / W)
    d.line([(0, i), (W, i)], fill=(14, 18 + c // 3, 30 + c))
# dos "fotogramas" y el intermedio
for k, (x, col) in enumerate([(70, (70, 120, 220)), (196, (120, 220, 255)), (322, (70, 120, 220))]):
    d.rounded_rectangle([x, 120, x + 120, 260], 14, outline=col, width=8 if k == 1 else 5)
d.polygon([(110, 230), (150, 160), (190, 230)], fill=(70, 120, 220))
d.polygon([(250, 230), (290, 160), (330, 230)], fill=(120, 220, 255))
d.polygon([(380, 230), (420, 160), (460, 230)], fill=(70, 120, 220))


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


d.text((W // 2, 340), "VCFG-NX", font=font(84), fill=(240, 245, 255), anchor="mm")
d.text((W // 2, 420), "Frame Gen + SMAA + FSR1", font=font(30), fill=(150, 200, 255), anchor="mm")
img.save(sys.argv[1] if len(sys.argv) > 1 else "sce_sys/icon0.png")
