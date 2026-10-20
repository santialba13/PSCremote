#include "ui_gfx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ui_font_data.h"

static ui_surf_t* T;
static int cx0, cy0, cx1, cy1;   // recorte [x0, x1) x [y0, y1)

void ui_target(ui_surf_t* s)
{
    T = s;
    ui_clip_all();
}

void ui_clip_all(void)
{
    cx0 = cy0 = 0;
    cx1 = T ? T->w : 0;
    cy1 = T ? T->h : 0;
}

void ui_clip(int x, int y, int w, int h)
{
    cx0 = x < 0 ? 0 : x;
    cy0 = y < 0 ? 0 : y;
    cx1 = x + w > T->w ? T->w : x + w;
    cy1 = y + h > T->h ? T->h : y + h;
}

// d + (s - d) * a / 255 por canal, con a en 0..255
static inline uint32_t blend(uint32_t d, uint32_t s, unsigned a)
{
    if (a >= 255) return s | 0xFF000000u;
    if (a == 0) return d;
    unsigned ia = 255 - a;
    unsigned rb = ((s & 0xFF00FFu) * a + (d & 0xFF00FFu) * ia + 0x800080u) >> 8;
    unsigned g = ((s & 0x00FF00u) * a + (d & 0x00FF00u) * ia + 0x008000u) >> 8;
    return 0xFF000000u | (rb & 0xFF00FFu) | (g & 0x00FF00u);
}

static inline uint32_t* row(int y) { return T->px + (size_t)y * T->stride; }

void ui_fill(int x, int y, int w, int h, uint32_t c)
{
    int x0 = x < cx0 ? cx0 : x, x1 = x + w > cx1 ? cx1 : x + w;
    int y0 = y < cy0 ? cy0 : y, y1 = y + h > cy1 ? cy1 : y + h;
    unsigned a = c >> 24;
    if (x0 >= x1 || y0 >= y1 || !a) return;
    for (int j = y0; j < y1; ++j) {
        uint32_t* p = row(j);
        if (a == 255)
            for (int i = x0; i < x1; ++i) p[i] = c;
        else
            for (int i = x0; i < x1; ++i) p[i] = blend(p[i], c, a);
    }
}

// cobertura (0..255) del pixel (px, py) en un rectangulo redondeado, solo
// para pixeles en las esquinas
static inline unsigned corner_cov(float px, float py, float ccx, float ccy, float r)
{
    float dx = px - ccx, dy = py - ccy;
    float d = sqrtf(dx * dx + dy * dy) - r;
    if (d <= -0.5f) return 255;
    if (d >= 0.5f) return 0;
    return (unsigned)((0.5f - d) * 255.0f + 0.5f);
}

static inline uint32_t lerp_color(uint32_t a, uint32_t b, int t, int n)
{
    if (n <= 0) return a;
    uint32_t out = 0;
    for (int sh = 0; sh < 32; sh += 8) {
        int ca = (a >> sh) & 0xFF, cb = (b >> sh) & 0xFF;
        out |= (uint32_t)(ca + (cb - ca) * t / n) << sh;
    }
    return out;
}

static void rrect_impl(int x, int y, int w, int h, int r, uint32_t top, uint32_t bottom)
{
    if (r * 2 > w) r = w / 2;
    if (r * 2 > h) r = h / 2;
    int y0 = y < cy0 ? cy0 : y, y1 = y + h > cy1 ? cy1 : y + h;
    int xa = x < cx0 ? cx0 : x, xb = x + w > cx1 ? cx1 : x + w;
    if (xa >= xb) return;
    for (int j = y0; j < y1; ++j) {
        uint32_t c = top == bottom ? top : lerp_color(top, bottom, j - y, h - 1);
        unsigned a = c >> 24;
        if (!a) continue;
        uint32_t* p = row(j);
        int in_top = j < y + r, in_bot = j >= y + h - r;
        if (!in_top && !in_bot) {
            for (int i = xa; i < xb; ++i) p[i] = blend(p[i], c, a);
            continue;
        }
        float ccy = in_top ? (float)(y + r) : (float)(y + h - r);
        float py = j + 0.5f;
        for (int i = xa; i < xb; ++i) {
            unsigned cov = 255;
            if (i < x + r) cov = corner_cov(i + 0.5f, py, (float)(x + r), ccy, (float)r);
            else if (i >= x + w - r) cov = corner_cov(i + 0.5f, py, (float)(x + w - r), ccy, (float)r);
            if (cov) p[i] = blend(p[i], c, cov * a / 255);
        }
    }
}

void ui_rrect(int x, int y, int w, int h, int r, uint32_t c) { rrect_impl(x, y, w, h, r, c, c); }
void ui_rrect_grad(int x, int y, int w, int h, int r, uint32_t top, uint32_t bottom)
{
    rrect_impl(x, y, w, h, r, top, bottom);
}

// cobertura de un rectangulo redondeado en un pixel cualquiera (lento: bordes)
static float rrect_cov(float px, float py, float x, float y, float w, float h, float r)
{
    float hx = w * 0.5f, hy = h * 0.5f;
    float qx = fabsf(px - (x + hx)) - (hx - r), qy = fabsf(py - (y + hy)) - (hy - r);
    float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
    float m = qx > qy ? qx : qy;
    float d = sqrtf(ox * ox + oy * oy) + (m < 0 ? m : 0) - r;
    return d <= -0.5f ? 1.0f : (d >= 0.5f ? 0.0f : 0.5f - d);
}

void ui_rrect_stroke(int x, int y, int w, int h, int r, int t, uint32_t c)
{
    unsigned a = c >> 24;
    int y0 = y < cy0 ? cy0 : y, y1 = y + h > cy1 ? cy1 : y + h;
    for (int j = y0; j < y1; ++j) {
        uint32_t* p = row(j);
        int band = j < y + r + t + 1 || j >= y + h - r - t - 1;
        int xa = x < cx0 ? cx0 : x, xb = x + w > cx1 ? cx1 : x + w;
        for (int i = xa; i < xb; ++i) {
            if (!band && i >= x + t + 1 && i < x + w - t - 1) { i = x + w - t - 2; continue; }
            float co = rrect_cov(i + 0.5f, j + 0.5f, (float)x, (float)y, (float)w, (float)h, (float)r);
            float ci = rrect_cov(i + 0.5f, j + 0.5f, (float)(x + t), (float)(y + t), (float)(w - 2 * t),
                                 (float)(h - 2 * t), (float)(r > t ? r - t : 0));
            float cov = co - ci;
            if (cov > 0.004f) p[i] = blend(p[i], c, (unsigned)(cov * a + 0.5f));
        }
    }
}

void ui_circle(int ccx, int ccy, int r, uint32_t c) { ui_rrect(ccx - r, ccy - r, 2 * r, 2 * r, r, c); }

void ui_copy(const ui_surf_t* src, int x, int y, int w, int h)
{
    int x0 = x < cx0 ? cx0 : x, x1 = x + w > cx1 ? cx1 : x + w;
    int y0 = y < cy0 ? cy0 : y, y1 = y + h > cy1 ? cy1 : y + h;
    if (x0 >= x1) return;
    for (int j = y0; j < y1; ++j)
        memcpy(row(j) + x0, src->px + (size_t)j * src->stride + x0, (size_t)(x1 - x0) * 4);
}

// ----------------------------------------------------------------- sprites
ui_img_t ui_img_new(int w, int h)
{
    ui_img_t im = { calloc((size_t)w * h, 4), w, h, NULL };
    if (!im.px) im.w = im.h = 0;
    return im;
}

void ui_img_free(ui_img_t* im)
{
    free(im->px);
    free(im->sp);
    im->px = NULL;
    im->sp = NULL;
    im->w = im->h = 0;
}

void ui_img_hole(ui_img_t* im, int x, int y, int w, int h, int r)
{
    for (int j = y; j < y + h && j < im->h; ++j) {
        int in_mid = j >= y + r && j < y + h - r;
        int a = in_mid ? x : x + r, b = in_mid ? x + w : x + w - r;
        for (int i = a < 0 ? 0 : a; i < b && i < im->w; ++i) im->px[j * im->w + i] = 0;
    }
}

void ui_img_spans(ui_img_t* im)
{
    free(im->sp);
    im->sp = malloc((size_t)im->h * 6 * sizeof(int16_t));
    if (!im->sp) return;
    for (int j = 0; j < im->h; ++j) {
        const uint32_t* p = im->px + (size_t)j * im->w;
        int16_t* s = im->sp + j * 6;
        int w = im->w, l0 = 0, l1, r0, r1 = w, runs = 0;
        while (l0 < w && !(p[l0] >> 24)) ++l0;
        while (r1 > l0 && !(p[r1 - 1] >> 24)) --r1;
        l1 = l0;
        while (l1 < r1 && (p[l1] >> 24)) ++l1;
        r0 = r1;
        while (r0 > l1 && (p[r0 - 1] >> 24)) --r0;
        for (int i = l1; i < r0; ++i) runs |= p[i] >> 24 != 0;
        if (runs) { l1 = r1; r0 = r1; }   // mas de dos tramos: uno solo, de punta a punta
        int o0 = 0, o1 = 0;
        for (int i = l0; i < r1;) {   // tramo opaco mas largo
            if ((p[i] >> 24) != 255) { ++i; continue; }
            int k = i;
            while (k < r1 && (p[k] >> 24) == 255) ++k;
            if (k - i > o1 - o0) { o0 = i; o1 = k; }
            i = k;
        }
        s[0] = (int16_t)l0; s[1] = (int16_t)l1; s[2] = (int16_t)r0; s[3] = (int16_t)r1;
        s[4] = (int16_t)o0; s[5] = (int16_t)o1;
    }
}

ui_img_t ui_make_shadow(int w, int h, int r, int blur, uint32_t c)
{
    ui_img_t im = ui_img_new(w + 2 * blur, h + 2 * blur);
    if (!im.px) return im;
    float hx = w * 0.5f, hy = h * 0.5f, a = (float)(c >> 24);
    for (int j = 0; j < im.h; ++j)
        for (int i = 0; i < im.w; ++i) {
            float qx = fabsf(i + 0.5f - im.w * 0.5f) - (hx - r), qy = fabsf(j + 0.5f - im.h * 0.5f) - (hy - r);
            float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
            float m = qx > qy ? qx : qy;
            float d = sqrtf(ox * ox + oy * oy) + (m < 0 ? m : 0) - r;   // distancia con signo al borde
            float t = d <= 0 ? 1.0f : 1.0f - d / blur;
            if (t <= 0) continue;
            t = t * t * (3 - 2 * t);   // suave
            im.px[j * im.w + i] = (c & 0xFFFFFFu) | (uint32_t)(a * t + 0.5f) << 24;
        }
    return im;
}

void ui_img_round(ui_img_t* im, int r)
{
    for (int j = 0; j < im->h; ++j)
        for (int i = 0; i < im->w; ++i) {
            if ((j >= r && j < im->h - r) || (i >= r && i < im->w - r)) continue;
            float cov = rrect_cov(i + 0.5f, j + 0.5f, 0, 0, (float)im->w, (float)im->h, (float)r);
            uint32_t p = im->px[j * im->w + i];
            im->px[j * im->w + i] = (p & 0xFFFFFFu) | (uint32_t)((p >> 24) * cov + 0.5f) << 24;
        }
}

static inline void blit_run(uint32_t* p, const uint32_t* s, int a0, int a1, int alpha)
{
    if (alpha == 255) {
        for (int i = a0; i < a1; ++i) {
            uint32_t c = s[i];
            unsigned a = c >> 24;
            if (a == 255) p[i] = c;
            else if (a) p[i] = blend(p[i], c, a);
        }
    } else {
        for (int i = a0; i < a1; ++i) {
            uint32_t c = s[i];
            unsigned a = (c >> 24) * (unsigned)alpha / 255;
            if (a) p[i] = blend(p[i], c, a);
        }
    }
}

// tramo [a0, a1) recortado a [x0, x1)
#define CLIPRUN(a0, a1) do { int u = (a0) < x0 ? x0 : (a0), v = (a1) > x1 ? x1 : (a1); \
    if (u < v) blit_run(p, s, u, v, alpha); } while (0)

void ui_blit(const ui_img_t* im, int x, int y, int alpha)
{
    if (!im->px) return;
    int x0 = x < cx0 ? cx0 : x, x1 = x + im->w > cx1 ? cx1 : x + im->w;
    int y0 = y < cy0 ? cy0 : y, y1 = y + im->h > cy1 ? cy1 : y + im->h;
    for (int j = y0; j < y1; ++j) {
        uint32_t* p = row(j);
        const uint32_t* s = im->px + (size_t)(j - y) * im->w - x;
        if (!im->sp) {
            CLIPRUN(x0, x1);
            continue;
        }
        const int16_t* sp = im->sp + (j - y) * 6;
        int l0 = x + sp[0], l1 = x + sp[1], r0 = x + sp[2], r1 = x + sp[3], o0 = x + sp[4], o1 = x + sp[5];
        if (alpha == 255 && o1 > o0) {   // la parte opaca se copia entera
            int u = o0 < x0 ? x0 : o0, v = o1 > x1 ? x1 : o1;
            if (u < v) memcpy(p + u, s + u, (size_t)(v - u) * 4);
            if (o0 >= l0 && o1 <= l1) { CLIPRUN(l0, o0); CLIPRUN(o1, l1); }
            else if (o0 >= r0 && o1 <= r1) { CLIPRUN(l0, l1); CLIPRUN(r0, o0); CLIPRUN(o1, r1); continue; }
            else { CLIPRUN(l0, l1); }
            if (r0 > l1 || r1 > l1) CLIPRUN(r0 > l1 ? r0 : l1, r1);
            continue;
        }
        CLIPRUN(l0, l1);
        if (r0 > l1 || r1 > l1) CLIPRUN(r0 > l1 ? r0 : l1, r1);
    }
}

// ------------------------------------------------------------------- texto
static const ui_glyph_t* glyph(const ui_font_t* f, unsigned char ch)
{
    if (ch < UI_FONT_FIRST) ch = '?';
    return &f->g[ch - UI_FONT_FIRST];
}

int ui_line_h(int font) { return ui_fonts[font].ascent + ui_fonts[font].descent; }

int ui_text_w(int font, const char* s)
{
    const ui_font_t* f = &ui_fonts[font];
    int w = 0;
    for (const unsigned char* t = (const unsigned char*)s; *t; ++t) w += glyph(f, *t)->adv;
    return w;
}

int ui_text(int font, int x, int y, uint32_t c, const char* s)
{
    const ui_font_t* f = &ui_fonts[font];
    unsigned ca = c >> 24;
    for (const unsigned char* t = (const unsigned char*)s; *t; ++t) {
        const ui_glyph_t* g = glyph(f, *t);
        int gx = x + g->x, gy = y + g->y;
        if (g->w && gx < cx1 && gx + g->w > cx0 && gy < cy1 && gy + g->h > cy0) {
            const uint8_t* b = f->bits + g->off;
            for (int j = 0; j < g->h; ++j) {
                int yy = gy + j;
                if (yy < cy0 || yy >= cy1) continue;
                uint32_t* p = row(yy);
                for (int i = 0; i < g->w; ++i) {
                    int xx = gx + i;
                    unsigned a = b[j * g->w + i];
                    if (a && xx >= cx0 && xx < cx1) p[xx] = blend(p[xx], c, a * ca / 255);
                }
            }
        }
        x += g->adv;
    }
    return x;
}

void ui_text_fit(int font, const char* s, int maxw, char* out, size_t n)
{
    size_t len = strlen(s);
    if (ui_text_w(font, s) <= maxw || n < 5) {
        snprintf(out, n, "%s", s);
        return;
    }
    int ell = ui_text_w(font, "...");
    const ui_font_t* f = &ui_fonts[font];
    int w = 0;
    size_t k = 0;
    while (k < len && k + 4 < n) {
        int a = glyph(f, (unsigned char)s[k])->adv;
        if (w + a + ell > maxw) break;
        w += a;
        ++k;
    }
    while (k > 0 && s[k - 1] == ' ') --k;
    memcpy(out, s, k);
    memcpy(out + k, "...", 4);
}

void ui_text_wrap2(int font, const char* s, int maxw, char* l1, char* l2, size_t n)
{
    l2[0] = 0;
    if (ui_text_w(font, s) <= maxw) {
        snprintf(l1, n, "%s", s);
        return;
    }
    // la mayor cantidad de palabras que entren en la primera linea
    size_t len = strlen(s), cut = 0;
    char tmp[256];
    for (size_t i = 1; i <= len && i < sizeof tmp; ++i) {
        if (s[i] != ' ' && s[i] != 0) continue;
        memcpy(tmp, s, i);
        tmp[i] = 0;
        if (ui_text_w(font, tmp) <= maxw) cut = i;
        else break;
    }
    if (!cut) {   // una palabra muy larga: cortar por caracteres
        ui_text_fit(font, s, maxw, l1, n);
        return;
    }
    snprintf(l1, n, "%.*s", (int)cut, s);
    const char* rest = s + cut;
    while (*rest == ' ') ++rest;
    ui_text_fit(font, rest, maxw, l2, n);
}

// ----------------------------------------------------------------- botones
int ui_button_size(void) { return UI_BTN_SIZE; }

void ui_button(int which, int x, int y)
{
    static const uint32_t* const sprites[4] = { ui_btn_cross, ui_btn_circle, ui_btn_square, ui_btn_triangle };
    ui_img_t im = { (uint32_t*)sprites[which & 3], UI_BTN_SIZE, UI_BTN_SIZE };
    ui_blit(&im, x, y, 255);
}
