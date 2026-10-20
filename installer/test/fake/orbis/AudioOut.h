#pragma once
#include <stdint.h>
enum OrbisAudioOutPort { ORBIS_AUDIO_OUT_PORT_TYPE_MAIN = 0 };
enum { ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_STEREO = 1 };
// en el PC no hay audio: la app sigue sin hilo de sonido
static inline int32_t sceAudioOutInit(void) { return 0; }
static inline int32_t sceAudioOutOpen(int u, enum OrbisAudioOutPort t, int32_t i, uint32_t len, uint32_t rate, uint32_t p) { return -1; }
static inline int32_t sceAudioOutOutput(int32_t h, const void* p) { return 0; }
static inline int32_t sceAudioOutClose(int32_t h) { return 0; }
