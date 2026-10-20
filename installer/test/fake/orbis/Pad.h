#pragma once
#include <stdint.h>
enum { ORBIS_PAD_BUTTON_OPTIONS = 0x8, ORBIS_PAD_BUTTON_UP = 0x10, ORBIS_PAD_BUTTON_RIGHT = 0x20, ORBIS_PAD_BUTTON_DOWN = 0x40,
       ORBIS_PAD_BUTTON_LEFT = 0x80, ORBIS_PAD_BUTTON_TRIANGLE = 0x1000, ORBIS_PAD_BUTTON_CIRCLE = 0x2000,
       ORBIS_PAD_BUTTON_CROSS = 0x4000, ORBIS_PAD_BUTTON_SQUARE = 0x8000 };
typedef struct { uint32_t buttons; } OrbisPadData;
static inline int scePadInit(void) { return 0; }
static inline int scePadOpen(int u, int t, int i, void* p) { return 1; }
static inline int scePadReadState(int h, OrbisPadData* d) { d->buttons = 0; return 0; }
