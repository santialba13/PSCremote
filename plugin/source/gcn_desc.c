#include "gcn_desc.h"

// IMG_DATA_FORMAT / IMG_NUM_FORMAT
#define DFMT_8_8_8_8      10
#define DFMT_2_10_10_10   9
#define NFMT_UNORM        0
#define NFMT_SRGB         9
// SQ_SEL_*
#define SEL_X 4
#define SEL_Y 5
#define SEL_Z 6
#define SEL_W 7
#define SQ_RSRC_IMG_2D 9

void nx_make_tsharp(uint32_t t[8], const nx_image_t* img)
{
    uint32_t dfmt = img->format == NX_FMT_RGB10A2_UNORM ? DFMT_2_10_10_10 : DFMT_8_8_8_8;
    uint32_t nfmt = img->format == NX_FMT_RGBA8_SRGB ? NFMT_SRGB : NFMT_UNORM;
    uint64_t base = img->gpu_addr >> 8;
    t[0] = (uint32_t)base;
    t[1] = (uint32_t)(base >> 32) & 0xFF;          // BASE_ADDRESS_HI, MIN_LOD = 0
    t[1] |= dfmt << 20 | nfmt << 26;
    t[2] = (img->width - 1) | (img->height - 1) << 14;
    // swizzle identidad: las escrituras de imagen ignoran DST_SEL en GCN, asi que
    // leer con swizzle y escribir sin el intercambiaria R y B.
    t[3] = SEL_X | SEL_Y << 3 | SEL_Z << 6 | SEL_W << 9;
    t[3] |= (img->tile_index & 0x1F) << 20;
    t[3] |= (uint32_t)SQ_RSRC_IMG_2D << 28;
    t[4] = (img->pitch - 1) << 13;                 // DEPTH = 0
    t[5] = 0;                                      // BASE_ARRAY / LAST_ARRAY = 0
    t[6] = 0;
    t[7] = 0;                                      // sin metadata (sin DCC/CMASK)
}

#define CLAMP_LAST_TEXEL 2
#define FILTER_POINT 0
#define FILTER_BILINEAR 1

void nx_make_ssharp(uint32_t s[4], int bilinear)
{
    uint32_t f = bilinear ? FILTER_BILINEAR : FILTER_POINT;
    s[0] = CLAMP_LAST_TEXEL | CLAMP_LAST_TEXEL << 3 | CLAMP_LAST_TEXEL << 6;
    s[0] |= 1u << 15;                              // FORCE_UNNORMALIZED
    s[1] = 0;
    s[2] = f << 20 | f << 22;                      // XY_MAG_FILTER / XY_MIN_FILTER, sin mips
    s[3] = 0;
}
