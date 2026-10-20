// Menu en pantalla (OSD) estilo VCFG y utilidades de imagen.
// El texto lo arma la CPU en una grilla de caracteres Latin-1 (cols x rows);
// el kernel dibuja el panel sobre la imagen de salida con la fuente 8x16.
#include "nx_common.h"

// Un work-item por pixel del panel. (x0, y0) esquina del panel en la imagen.
// sel_row: fila resaltada (-1 = ninguna). Fondo semitransparente.
NX_KERNEL void osd_draw(tdesc_t dst, int W, int H,
                        __global const uchar* text, int cols, int rows,
                        __global const uchar* font, int x0, int y0, int scale, int sel_row)
{
    const int pad = 6 * scale;
    int pw = cols * 8 * scale + 2 * pad, ph = rows * 16 * scale + 2 * pad;
    int lx = GID_X, ly = GID_Y;
    if (lx >= pw || ly >= ph) return;
    int x = x0 + lx, y = y0 + ly;
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    float4 c = nx_load(dst, x, y);
    int tx = (lx - pad) / scale, ty = (ly - pad) / scale;
    int col = tx >> 3, row = ty >> 4;
    float4 bg = (float4)(0.04f, 0.06f, 0.10f, 1.0f);
    float alpha = 0.82f;
    bool inside = lx >= pad && ly >= pad && col < cols && row < rows;
    if (inside && row == sel_row) { bg = (float4)(0.10f, 0.30f, 0.62f, 1.0f); alpha = 0.92f; }
    // borde de 2 px
    if (lx < 2 || ly < 2 || lx >= pw - 2 || ly >= ph - 2) { bg = (float4)(0.35f, 0.62f, 1.0f, 1.0f); alpha = 1.0f; }
    c = c * (1.0f - alpha) + bg * alpha;
    if (inside) {
        int ch = text[row * cols + col];
        if (ch >= 32) {
            uchar bits = font[(ch - 32) * 16 + (ty & 15)];
            if ((bits >> (7 - (tx & 7))) & 1)
                c = (float4)(0.95f, 0.97f, 1.0f, 1.0f);
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
