// Sonidos de la interfaz, generados por codigo (48 kHz, estereo, 16 bits).
// ui_sound_play() se llama desde el hilo de la interfaz; ui_sound_mix() desde
// el hilo de audio (cola de un productor y un consumidor, sin bloqueos).
#pragma once
#include <stdint.h>

enum { SND_MOVE, SND_TAB, SND_SELECT, SND_BACK, SND_ERROR, SND_SAVE, SND_COUNT };

#define UI_SOUND_RATE 48000

void ui_sound_init(void);            // genera las muestras
void ui_sound_play(int id);
int ui_sound_busy(void);             // hay algo sonando o por sonar
void ui_sound_mix(int16_t* out, int frames);   // frames estereo intercalados
