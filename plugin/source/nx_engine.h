// Motor de VCFG-NX: toda la logica de decision del plugin, sin llamadas
// directas al sistema. main.c le da una "plataforma" con las funciones reales
// de la PS4; plugin/test/console_sim.c le da una consola simulada para probar
// en el PC que ningun fallo (memoria, GPU trabada, cierre del juego) llegue a
// la consola.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "config.h"

typedef struct {
    int32_t format, tmode, aspect;
    uint32_t width, height, pitch;
} nx_vo_attr_t;

typedef struct {
    uint32_t count;
    void** dcb;
    uint32_t* dcb_sz;
    void** ccb;
    uint32_t* ccb_sz;
    uint32_t handle, index, mode;
    int64_t arg;
    uint64_t workload;     // solo para las variantes ForWorkload
    int for_workload;
    int cpu_flip;          // sceVideoOutSubmitFlip: count = 0, el juego ya envio su trabajo
} nx_submit_t;

typedef struct {
    void* (*alloc_gpu)(size_t len, size_t align);           // NULL si falla
    size_t (*avail_direct)(void);                           // memoria directa libre (bytes)
    int (*register_buffers)(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a);
    int (*set_flip_rate)(int handle, int rate);
    int (*submit)(const nx_submit_t* s);                    // DCB/CCB de s, sin flip
    int (*submit_flip)(const nx_submit_t* s);               // DCB/CCB de s + flip (funcion original)
    uint64_t (*now_us)(void);
    void (*sleep_us)(uint32_t us);
    void (*notify)(const char* msg);
    void (*log)(const char* msg);
    int (*read_file)(const char* path, char* buf, int cap); // bytes leidos o < 0
    int (*write_file)(const char* path, const char* data, int len);
} nx_platform_t;

typedef enum {
    NX_ST_OFF = 0,       // pasa todo intacto (formato no soportado, proceso no compatible...)
    NX_ST_WAIT_BUFFERS,  // esperando que el juego registre sus framebuffers
    NX_ST_WARMUP,        // juego arrancando: no se reserva nada todavia
    NX_ST_RUN,           // listo (con o sin memoria para los efectos)
    NX_ST_DEGRADED,      // la GPU no da abasto: solo pasa los fotogramas del juego
} nx_state_t;

typedef struct {
    nx_state_t state;
    int full_ready;       // arena y buffers de salida reservados
    int safe_mode;        // la sesion anterior se cerro con el plugin activo
    int menu_open, menu_row;
    float game_fps;
    uint64_t flips, our_jobs, passthrough, stalls, alloc_fail;
    char status[64];
    const uint8_t* menu_text;   // grilla del menu (menu_cols x menu_rows, Latin-1), para pruebas
    int menu_cols, menu_rows;
} nx_engine_info_t;

void nx_engine_init(const nx_platform_t* pf, const char* title_id);
void nx_engine_on_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a, int rc);
int nx_engine_on_set_flip_rate(int handle, int rate);      // devuelve la tasa a aplicar
int nx_engine_on_submit_flip(const nx_submit_t* s);         // hace las submits; devuelve el rc del juego
// lectura del mando: abre/cierra el menu con L3+R3 y, con el menu abierto,
// se queda con los botones (el juego recibe "nada pulsado")
void nx_engine_on_pad(uint32_t* buttons, uint8_t sticks[4], uint8_t triggers[2]);
const nx_engine_info_t* nx_engine_info(void);
