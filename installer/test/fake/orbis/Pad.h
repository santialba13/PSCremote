#pragma once
#include <stdint.h>
enum { ORBIS_PAD_BUTTON_OPTIONS = 0x8, ORBIS_PAD_BUTTON_L1 = 0x400, ORBIS_PAD_BUTTON_R1 = 0x800,
       ORBIS_PAD_BUTTON_L2 = 0x100, ORBIS_PAD_BUTTON_R2 = 0x200, ORBIS_PAD_BUTTON_UP = 0x10, ORBIS_PAD_BUTTON_RIGHT = 0x20, ORBIS_PAD_BUTTON_DOWN = 0x40,
       ORBIS_PAD_BUTTON_LEFT = 0x80, ORBIS_PAD_BUTTON_TRIANGLE = 0x1000, ORBIS_PAD_BUTTON_CIRCLE = 0x2000,
       ORBIS_PAD_BUTTON_CROSS = 0x4000, ORBIS_PAD_BUTTON_SQUARE = 0x8000 };
typedef struct { uint32_t buttons; } OrbisPadData;
static inline int scePadInit(void) { return 0; }
static inline int scePadOpen(int u, int t, int i, void* p) { return 1; }
// el test puede guionar el mando: fake_pad[] con un estado de botones por lectura
static const uint32_t* fake_pad;
static int fake_pad_n, fake_pad_i;
static inline int scePadReadState(int h, OrbisPadData* d)
{
    d->buttons = fake_pad && fake_pad_i < fake_pad_n ? fake_pad[fake_pad_i] : 0;
    ++fake_pad_i;
    extern void fake_pad_exhausted(void);
    if (fake_pad && fake_pad_i > fake_pad_n + 200) fake_pad_exhausted();   // el guion termino sin salir
    return 0;
}
