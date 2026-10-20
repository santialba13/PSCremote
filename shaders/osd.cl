// Menu en pantalla (OSD) estilo VCFG y utilidades de imagen.
// El texto lo arma la CPU en una grilla de caracteres Latin-1 (cols x rows) con
// un color por celda; el kernel dibuja el panel redondeado y translucido sobre la
// imagen de salida con la fuente suavizada de nx_osd_font.h.
#include "nx_common.h"

// Cobertura (0..1) de un rectangulo redondeado [0,w) x [0,h) de radio r en (px, py)
static float rr_cov(float px, float py, float w, float h, float r)
{
    float hx = w * 0.5f, hy = h * 0.5f;
    float qx = nx_absf(px - hx) - (hx - r), qy = nx_absf(py - hy) - (hy - r);
    float ox = nx_maxf(qx, 0.0f), oy = nx_maxf(qy, 0.0f);
    float d = __builtin_sqrtf(ox * ox + oy * oy) + nx_minf(nx_maxf(qx, qy), 0.0f) - r;
    return nx_clampf(0.5f - d, 0.0f, 1.0f);
}

// paleta de los atributos de cada celda
static float4 palette(int a)
{
    switch (a) {
    case 1: return (float4)(0.64f, 0.69f, 0.80f, 1.0f);   // atenuado
    case 2: return (float4)(0.20f, 0.83f, 0.60f, 1.0f);   // ON (verde)
    case 3: return (float4)(0.45f, 0.49f, 0.60f, 1.0f);   // OFF (gris)
    case 4: return (float4)(0.40f, 0.62f, 1.00f, 1.0f);   // acento (azul)
    case 5: return (float4)(1.00f, 0.71f, 0.28f, 1.0f);   // aviso (ambar)
    case 6: return (float4)(1.00f, 1.00f, 1.00f, 1.0f);   // titulo
    case 7: return (float4)(1.00f, 0.37f, 0.42f, 1.0f);   // error (rojo)
    default: return (float4)(0.95f, 0.96f, 1.00f, 1.0f);
    }
}

// Un work-item por pixel del panel. (x0, y0) esquina del panel en la imagen.
// text/attr: grilla cols x rows (Latin-1 y color); font: atlas de alfa con celdas
// cw x ch; scale: 1 (1080p) o 2 (4K). sel_row: fila resaltada (-1 = ninguna).
NX_KERNEL void osd_draw(tdesc_t dst, int W, int H,
                        __global const uchar* text, __global const uchar* attr, int cols, int rows,
                        __global const uchar* font, int cw, int ch, int x0, int y0, int scale, int sel_row)
{
    const int pad = 18 * scale;
    int pw = cols * cw * scale + 2 * pad, ph = rows * ch * scale + 2 * pad;
    int lx = GID_X, ly = GID_Y;
    if (lx >= pw || ly >= ph) return;
    int x = x0 + lx, y = y0 + ly;
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    float px = lx + 0.5f, py = ly + 0.5f, r = 18.0f * scale;
    float cov = rr_cov(px, py, (float)pw, (float)ph, r);
    if (cov <= 0.0f) return;   // fuera de las esquinas: el juego intacto
    float4 c = nx_load(dst, x, y);
    // fondo translucido con un leve degradado y borde de luz
    float t = (float)ly / ph;
    float4 bg = (float4)(0.055f + 0.02f * t, 0.075f + 0.025f * t, 0.14f + 0.04f * t, 1.0f);
    c = c + (bg - c) * (0.86f * cov);
    float ring = cov - rr_cov(px - 1.5f * scale, py - 1.5f * scale, pw - 3.0f * scale, ph - 3.0f * scale,
                              r - 1.5f * scale);
    c = c + ((float4)(0.40f, 0.62f, 1.0f, 1.0f) - c) * (0.55f * nx_clampf(ring, 0.0f, 1.0f));
    // fila elegida: pastilla azul que sobresale 8 px a cada lado del texto
    if (sel_row >= 0 && ly >= pad + sel_row * ch * scale && ly < pad + (sel_row + 1) * ch * scale) {
        float hx = lx - (pad - 8 * scale) + 0.5f, hy = ly - (pad + sel_row * ch * scale) + 0.5f;
        float hc = rr_cov(hx, hy, (float)(cols * cw * scale + 16 * scale), (float)(ch * scale), 9.0f * scale);
        c = c + ((float4)(0.22f, 0.42f, 0.86f, 1.0f) - c) * (0.75f * hc);
    }
    int tx = (lx - pad) / scale, ty = (ly - pad) / scale;
    if (lx >= pad && ly >= pad && tx < cols * cw && ty < rows * ch) {
        int col = tx / cw, row = ty / ch;
        int chr = text[row * cols + col];
        if (chr >= 32) {
            float a = font[((chr - 32) * ch + (ty - row * ch)) * cw + (tx - col * cw)] * (1.0f / 255.0f);
            if (a > 0.0f) c = c + (palette(attr[row * cols + col]) - c) * a;
        }
    }
    c.w = 1.0f;
    nx_store(dst, x, y, c);
}

// Copia el rectangulo [x0, x0+w) x [y0, y0+h) de src a dst (mismo tamano):
// el modo "comparar" deja la mitad izquierda sin procesar.
NX_KERNEL void copy_rect(tdesc_t src, tdesc_t dst, int x0, int y0, int w, int h)
{
    int x = GID_X, y = GID_Y;
    if (x >= w || y >= h) return;
    nx_store(dst, x0 + x, y0 + y, nx_load(src, x0 + x, y0 + y));
}
