// Motor de VCFG-NX: toda la logica de decision del plugin, sin llamadas
// directas al sistema. main.c le da una "plataforma" con las funciones reales
// de la PS4; plugin/test/console_sim.c le da una consola simulada para probar
// en el PC que ningun fallo (memoria, GPU trabada, cierre del juego) llegue a
// la consola.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "config.h"

// registros y estado de arranque: en la carpeta de GoldHEN (GoldHEN la lee y
// escribe desde dentro de los juegos; /data/vcfg_nx no siempre se pudo escribir)
#define NX_LOGDIR "/data/GoldHEN/vcfg_nx"

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
    int (*append_file)(const char* path, const char* data, int len);   // registro (puede ser NULL)
    int (*is_pro)(void);                                    // 1 = el juego corre en modo PS4 Pro (puede ser NULL)
    int (*temp_c)(void);                                    // temperatura del procesador en grados C, < 0 si no se sabe (puede ser NULL)
    size_t (*pool_free)(int base);                          // bytes libres de la reserva temprana (base = 1: la del menu); puede ser NULL
    void (*stage)(const char* msg);                         // hito anotado al instante (traza); puede ser NULL
    // memoria que escribe la CPU y lee la GPU (WB_ONION, coherente; tablas y DCBs de la
    // generacion VCFG); NULL = usar alloc_gpu
    void* (*alloc_shared)(size_t len, size_t align);
    // sceGnmInsertWaitFlipDone: 7 dwords en cmd para que la GPU espere a que el buffer
    // no este en pantalla; 0 = ok. NULL = no hay
    int (*wait_flip_done)(uint32_t* cmd, uint32_t ndw, int32_t handle, uint32_t index);
} nx_platform_t;

typedef enum {
    NX_ST_OFF = 0,       // pasa todo intacto (formato no soportado, proceso no compatible...)
    NX_ST_WAIT_BUFFERS,  // esperando que el juego registre sus framebuffers
    NX_ST_WARMUP,        // juego arrancando: no se reserva nada todavia
    NX_ST_RUN,           // listo (con o sin memoria para los efectos)
    NX_ST_DEGRADED,      // la GPU no da abasto ni sin efectos, o algo inesperado: solo pasa los fotogramas
} nx_state_t;

typedef struct {
    nx_state_t state;
    int full_ready;       // arena y buffers de salida reservados
    int menu_x, menu_y, menu_w, menu_h;   // donde se dibujo el menu la ultima vez (px)
    float gpu_ms;         // tiempo de GPU de nuestro trabajo por fotograma (promedio)
    int safe_mode;        // la sesion anterior se cerro con el plugin activo
    int menu_open, menu_row;
    float game_fps;
    int level, level_max;   // calidad dinamica: escalones bajados / posibles (0 = como se configuro)
    int level_locked;       // las subidas fallaron varias veces: espera el maximo entre intentos
    int hq;                 // NXSR Calidad en uso
    int temp_c;             // ultima temperatura leida (-1 = desconocida)
    int hot;                // 1 = bajando por temperatura, 2 = temperatura critica (todo apagado)
    uint64_t readopts;      // re-registros de buffers del juego adoptados sin apagarse
    uint64_t flips, our_jobs, passthrough, stalls, alloc_fail;
    char status[64];
    const uint8_t* menu_text;   // grilla del menu (menu_cols x menu_rows, Latin-1), para pruebas
    const uint8_t* menu_attr;   // color de cada celda
    int menu_cols, menu_rows;
} nx_engine_info_t;

void nx_engine_init(const nx_platform_t* pf, const char* title_id);
// tope de efectos del arranque escalonado: 0 solo menu/contador, 1 nitidez
// (RCAS), 2 + reescalado IA y SMAA, 3 + generacion de fotogramas (todo). Por defecto 3.
void nx_engine_allow_effects(int cap);
void nx_engine_on_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a, int rc);
int nx_engine_on_set_flip_rate(int handle, int rate);      // devuelve la tasa a aplicar
int nx_engine_on_submit_flip(const nx_submit_t* s);         // hace las submits; devuelve el rc del juego
// lectura del mando: abre/cierra el menu con L3+R3 y, con el menu abierto,
// se queda con los botones (el juego recibe "nada pulsado")
void nx_engine_on_pad(uint32_t* buttons, uint8_t sticks[4], uint8_t triggers[2]);
// hilo de sondeo propio del plugin (no depende de como lea el mando el juego):
// con el sondeo activo los hooks solo filtran los botones con el menu abierto
void nx_engine_pad_logic(uint32_t buttons);
void nx_engine_pad_polled(int on);
// vuelca el registro pendiente a /data/vcfg_nx/logs/<TITLE>.log
void nx_engine_flush_log(void);
void nx_engine_tick(void);
void nx_engine_note(const char* line);   // una linea al registro del juego (desde la plataforma)   // ~1 vez por segundo desde el hilo del plugin (diagnostico)
const nx_engine_info_t* nx_engine_info(void);
