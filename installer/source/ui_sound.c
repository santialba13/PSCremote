#include "ui_sound.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int16_t* pcm; int len; } snd_t;
static snd_t sounds[SND_COUNT];

#define MAX_VOICES 4
static struct { int id, pos; } voices[MAX_VOICES];
static int nvoices;
static volatile int queue[16];
static volatile unsigned q_head, q_tail;   // head: lo escribe play; tail: lo escribe mix

// nota con ataque corto y caida exponencial; f1 != f0 hace un barrido
static void tone(float* buf, int len, int at, float f0, float f1, float dur, float amp, float decay, float harm)
{
    int n = (int)(dur * UI_SOUND_RATE);
    float ph = 0;
    for (int i = 0; i < n && at + i < len; ++i) {
        float t = (float)i / UI_SOUND_RATE, k = (float)i / n;
        float f = f0 + (f1 - f0) * k;
        ph += 2.0f * 3.14159265f * f / UI_SOUND_RATE;
        float env = (t < 0.002f ? t / 0.002f : 1.0f) * expf(-t / decay);
        env *= 1.0f - k * k * k;   // sin click al final
        buf[at + i] += amp * env * (sinf(ph) + harm * sinf(2 * ph) + harm * 0.5f * sinf(3 * ph));
    }
}

static void make(int id, float dur)
{
    int len = (int)(dur * UI_SOUND_RATE);
    float* f = calloc((size_t)len, sizeof *f);
    int16_t* p = malloc((size_t)len * sizeof *p);
    if (!f || !p) { free(f); free(p); return; }
    switch (id) {
    case SND_MOVE:   tone(f, len, 0, 1850, 1850, 0.040f, 0.16f, 0.010f, 0.10f); break;
    case SND_TAB:    tone(f, len, 0, 1320, 1500, 0.050f, 0.17f, 0.014f, 0.12f); break;
    case SND_SELECT:
        tone(f, len, 0, 988, 988, 0.050f, 0.20f, 0.020f, 0.15f);
        tone(f, len, (int)(0.045f * UI_SOUND_RATE), 1480, 1480, 0.090f, 0.20f, 0.030f, 0.15f);
        break;
    case SND_BACK:   tone(f, len, 0, 820, 520, 0.085f, 0.18f, 0.030f, 0.12f); break;
    case SND_ERROR:
        tone(f, len, 0, 196, 190, 0.090f, 0.20f, 0.060f, 0.45f);
        tone(f, len, (int)(0.11f * UI_SOUND_RATE), 185, 180, 0.120f, 0.20f, 0.070f, 0.45f);
        break;
    case SND_SAVE:
        tone(f, len, 0, 784, 784, 0.25f, 0.16f, 0.080f, 0.10f);
        tone(f, len, (int)(0.07f * UI_SOUND_RATE), 988, 988, 0.25f, 0.16f, 0.080f, 0.10f);
        tone(f, len, (int)(0.14f * UI_SOUND_RATE), 1319, 1319, 0.30f, 0.17f, 0.100f, 0.10f);
        break;
    }
    for (int i = 0; i < len; ++i) {
        float v = f[i] * 1.35f;
        v = v > 1 ? 1 : (v < -1 ? -1 : v);
        p[i] = (int16_t)(v * 32000);
    }
    free(f);
    sounds[id].pcm = p;
    sounds[id].len = len;
}

void ui_sound_init(void)
{
    static const float dur[SND_COUNT] = { 0.045f, 0.055f, 0.14f, 0.09f, 0.24f, 0.46f };
    for (int i = 0; i < SND_COUNT; ++i)
        if (!sounds[i].pcm) make(i, dur[i]);
}

void ui_sound_play(int id)
{
    if (id < 0 || id >= SND_COUNT) return;
    unsigned h = q_head;
    if (h - q_tail >= 16) return;   // cola llena: se pierde este sonido
    queue[h & 15] = id;
    q_head = h + 1;
}

int ui_sound_busy(void) { return nvoices > 0 || q_head != q_tail; }

void ui_sound_mix(int16_t* out, int frames)
{
    while (q_tail != q_head) {
        int id = queue[q_tail & 15];
        q_tail = q_tail + 1;
        if (id == SND_MOVE || id == SND_TAB)   // al mantener la flecha: reinicia en vez de apilar
            for (int v = 0; v < nvoices; ++v)
                if (voices[v].id == id) voices[v] = voices[--nvoices];
        if (nvoices == MAX_VOICES) voices[0] = voices[--nvoices];
        voices[nvoices].id = id;
        voices[nvoices].pos = 0;
        ++nvoices;
    }
    for (int i = 0; i < frames; ++i) {
        int acc = 0;
        for (int v = 0; v < nvoices; ++v) {
            const snd_t* s = &sounds[voices[v].id];
            if (voices[v].pos < s->len) acc += s->pcm[voices[v].pos++];
        }
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;
        out[2 * i] = out[2 * i + 1] = (int16_t)acc;
    }
    for (int v = 0; v < nvoices;)
        if (voices[v].pos >= sounds[voices[v].id].len) voices[v] = voices[--nvoices];
        else ++v;
}
