// Genera cada sonido de la app a WAV (para escucharlos en el PC) y revisa
// duracion, nivel, que no sature y que no quede "click" al final.
//   cc -I../source test_sound.c ../source/ui_sound.c -lm && ./a.out carpeta
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ui_sound.h"

static void wav(const char* path, const int16_t* st, int frames)
{
    FILE* f = fopen(path, "wb");
    int data = frames * 4, rate = UI_SOUND_RATE, br = rate * 4;
    fwrite("RIFF", 1, 4, f); int v = 36 + data; fwrite(&v, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f);
    v = 16; fwrite(&v, 4, 1, f); short s = 1; fwrite(&s, 2, 1, f); s = 2; fwrite(&s, 2, 1, f);
    fwrite(&rate, 4, 1, f); fwrite(&br, 4, 1, f); s = 4; fwrite(&s, 2, 1, f); s = 16; fwrite(&s, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f); fwrite(st, 2, (size_t)frames * 2, f);
    fclose(f);
}

int main(int argc, char** argv)
{
    static const char* names[SND_COUNT] = { "mover", "pestana", "elegir", "volver", "error", "guardado" };
    int fails = 0;
    ui_sound_init();
    static int16_t buf[UI_SOUND_RATE * 2];
    for (int id = 0; id < SND_COUNT; ++id) {
        ui_sound_play(id);
        int n = 0;
        while (ui_sound_busy() && n + 256 <= UI_SOUND_RATE) { ui_sound_mix(buf + 2 * n, 256); n += 256; }
        int peak = 0;
        for (int i = 0; i < n; ++i) { int a = abs(buf[2 * i]); if (a > peak) peak = a; }
        int tail = abs(buf[2 * (n - 1)]);
        printf("%-9s %4d ms  pico %5.1f%%  final %d\n", names[id], n * 1000 / UI_SOUND_RATE, peak * 100.0 / 32767, tail);
        if (peak > 32000 || peak < 2000 || tail > 200) { printf("  FALLO\n"); ++fails; }
        if (argc > 1) { char p[256]; snprintf(p, sizeof p, "%s/%s.wav", argv[1], names[id]); wav(p, buf, n); }
    }
    // mantener la flecha: 10 "mover" seguidos no se apilan (no satura)
    for (int k = 0; k < 10; ++k) ui_sound_play(SND_MOVE);
    ui_sound_mix(buf, 256);
    int peak = 0;
    for (int i = 0; i < 256; ++i) if (abs(buf[2 * i]) > peak) peak = abs(buf[2 * i]);
    if (peak > 12000) { printf("FALLO: los sonidos de mover se apilan (pico %d)\n", peak); ++fails; }
    while (ui_sound_busy()) ui_sound_mix(buf, 256);
    printf(fails ? "%d FALLOS\n" : "OK: sonidos\n", fails);
    return fails != 0;
}
