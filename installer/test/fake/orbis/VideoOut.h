#pragma once
#include <stdint.h>
#define ORBIS_VIDEO_USER_MAIN 0xFF
#define ORBIS_VIDEO_OUT_BUS_MAIN 0
#define ORBIS_VIDEO_OUT_FLIP_VSYNC 1
typedef struct { int32_t format, tmode, aspect; uint32_t width, height, pixelPitch; uint64_t reserved[2]; } OrbisVideoOutBufferAttribute;
static inline int sceVideoOutOpen(int u, int b, int i, const void* p) { return 1; }
static inline void sceVideoOutSetBufferAttribute(void* a, uint32_t f, uint32_t t, uint32_t asp, uint32_t w, uint32_t h, uint32_t p) {}
static inline int sceVideoOutSetFlipRate(int h, int r) { return 0; }
static inline int sceVideoOutRegisterBuffers(int h, int s, void* const* a, int n, const OrbisVideoOutBufferAttribute* at) { return 0; }
static inline int sceVideoOutSubmitFlip(int h, int i, uint32_t m, int64_t a) { return 0; }
