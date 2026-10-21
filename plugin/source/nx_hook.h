// Enganche de funciones propio (reemplaza HOOK32 del SDK de GoldHEN).
//
// HOOK32 del SDK pone el puente hacia nuestro codigo en memoria de malloc y
// salta ahi con un jmp de 32 bits (+-2 GB): si ese malloc queda lejos de la
// funcion, el salto cae en cualquier lado y el juego se cierra la primera vez
// que la llama. HOOK (64 bits) copia 14 bytes de la funcion sin corregir las
// instrucciones relativas. Aca:
//   * la memoria del puente se pide ejecutable y CERCA de la funcion (se
//     verifica la distancia antes de usarla);
//   * las instrucciones copiadas se decodifican (HDE64) y las relativas se
//     corrigen; si alguna no se puede corregir, no se engancha (y se dice por que);
//   * se siguen los "thunks" (jmp rel32 / jmp [rip]) hasta la funcion real.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    void* target;      // funcion enganchada (despues de seguir thunks)
    void* stub;        // llamar aca = llamar a la original
    uint8_t* mem;      // pagina del puente (cerca de target)
    uint8_t saved[32]; // bytes originales
    int nsaved;
    char why[80];      // si fallo: motivo
} nx_hook_t;

// 0 = enganchada; < 0 = no se toco nada (h->why dice por que)
int nx_hook_install(nx_hook_t* h, void* target, void* hook);
void nx_hook_remove(nx_hook_t* h);

// plataforma (main.c / prueba en PC): memoria ejecutable cerca de `near`
// (NULL si no se pudo) y permisos de escritura sobre codigo
void* nx_hook_map_near(void* near, size_t len);
void nx_hook_unmap(void* p, size_t len);
int nx_hook_make_writable(void* p, size_t len);
