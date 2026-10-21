#include "nx_osd.h"

#include <string.h>

#include "../include/nx_osd_font.h"

#define RING_BYTES 8192

typedef struct { uint64_t dst; int32_t W, H; uint64_t text, attr; int32_t cols, rows; uint64_t font;
                 int32_t cw, ch, x0, y0, scale, sel_row; } ka_osd_t;

void nx_osd_panel_size(int cols, int rows, int scale, int* w, int* h)
{
    *w = cols * NX_OSD_CW * scale + 36 * scale;
    *h = rows * NX_OSD_CH * scale + 36 * scale;
}
typedef struct { uint64_t src, dst; int32_t x0, y0, w, h; } ka_copy_t;

size_t nx_osd_mem_size(void)
{
    const nx_shader_t* a = nx_find_shader("osd_draw");
    const nx_shader_t* b = nx_find_shader("copy_rect");
    size_t code = (a ? a->code_dwords * 4 : 0) + (b ? b->code_dwords * 4 : 0);
    return (code + 512 + 255) / 256 * 256 + 2 * RING_BYTES;
}

int nx_osd_init(nx_osd_ctx_t* o, uint8_t* cpu, uint64_t gpu)
{
    memset(o, 0, sizeof *o);
    o->sh_draw = nx_find_shader("osd_draw");
    o->sh_copy = nx_find_shader("copy_rect");
    if (!o->sh_draw || !o->sh_copy) return -1;
    size_t off = 0;
    memcpy(cpu + off, o->sh_draw->code, o->sh_draw->code_dwords * 4);
    o->code_draw = gpu + off;
    off = (off + o->sh_draw->code_dwords * 4 + 255) / 256 * 256;
    memcpy(cpu + off, o->sh_copy->code, o->sh_copy->code_dwords * 4);
    o->code_copy = gpu + off;
    off = (off + o->sh_copy->code_dwords * 4 + 255) / 256 * 256;
    for (int s = 0; s < 2; ++s) {
        o->ring[s] = (nx_ring_t){ cpu + off, gpu + off, RING_BYTES, 0 };
        off += RING_BYTES;
    }
    return 0;
}

void nx_osd_begin(nx_osd_ctx_t* o, int slot) { o->ring[slot].used = 0; }

static uint64_t tsharp(nx_ring_t* r, const nx_image_t* img)
{
    uint32_t t[8];
    nx_make_tsharp(t, img);
    return nx_ring_put(r, t, sizeof t, 32);
}

int nx_osd_draw(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* img, const nx_osd_t* const* osds, int n)
{
    for (int i = 0; i < n; ++i) {
        const nx_osd_t* d = osds[i];
        if (!d) continue;
        int pw, ph;
        nx_osd_panel_size(d->cols, d->rows, d->scale, &pw, &ph);
        uint64_t t = tsharp(&o->ring[slot], img);
        if (!t) return -1;
        ka_osd_t a = { t, (int32_t)img->width, (int32_t)img->height, d->text_gpu, d->attr_gpu, d->cols, d->rows,
                       d->font_gpu, NX_OSD_CW, NX_OSD_CH, d->x0, d->y0, d->scale, d->sel_row };
        int r = nx_dispatch(c, &o->ring[slot], o->sh_draw, o->code_draw, &a, sizeof a, pw, ph);
        if (r) return r;
    }
    return 0;
}

int nx_osd_copy_left(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst)
{
    int w = (int)(src->width < dst->width ? src->width : dst->width) / 2;
    int h = (int)(src->height < dst->height ? src->height : dst->height);
    uint64_t ts = tsharp(&o->ring[slot], src), td = tsharp(&o->ring[slot], dst);
    if (!ts || !td) return -1;
    ka_copy_t a = { ts, td, 0, 0, w, h };
    return nx_dispatch(c, &o->ring[slot], o->sh_copy, o->code_copy, &a, sizeof a, w, h);
}

int nx_osd_copy(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst)
{
    int w = (int)(src->width < dst->width ? src->width : dst->width);
    int h = (int)(src->height < dst->height ? src->height : dst->height);
    uint64_t ts = tsharp(&o->ring[slot], src), td = tsharp(&o->ring[slot], dst);
    if (!ts || !td) return -1;
    ka_copy_t a = { ts, td, 0, 0, w, h };
    return nx_dispatch(c, &o->ring[slot], o->sh_copy, o->code_copy, &a, sizeof a, w, h);
}

int nx_osd_copy_rect(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst, int x0, int y0,
                     int w, int h)
{
    int mw = (int)(src->width < dst->width ? src->width : dst->width);
    int mh = (int)(src->height < dst->height ? src->height : dst->height);
    if (x0 < 0) { w += x0; x0 = 0; }
    if (y0 < 0) { h += y0; y0 = 0; }
    if (x0 + w > mw) w = mw - x0;
    if (y0 + h > mh) h = mh - y0;
    if (w <= 0 || h <= 0) return 0;
    uint64_t ts = tsharp(&o->ring[slot], src), td = tsharp(&o->ring[slot], dst);
    if (!ts || !td) return -1;
    ka_copy_t a = { ts, td, x0, y0, w, h };
    return nx_dispatch(c, &o->ring[slot], o->sh_copy, o->code_copy, &a, sizeof a, w, h);
}
