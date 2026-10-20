#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
typedef struct { int type, reqId, priority, msgId, targetId, userId, unk1, unk2, appId, errorNum, unk3; char useIconImageUri; char message[1024]; char iconUri[1024]; char unk[1024]; } OrbisNotificationRequest;
enum { NotificationRequest = 0 };
static inline int sceKernelSendNotificationRequest(int a, OrbisNotificationRequest* r, size_t s, int b) { return 0; }
static uint64_t fake_clock_us;
static inline int sceKernelUsleep(unsigned u) { fake_clock_us += u; return 0; }
static inline uint64_t sceKernelGetProcessTime(void) { return fake_clock_us; }
static inline size_t sceKernelGetDirectMemorySize(void) { return 1u << 30; }
static inline int sceKernelAllocateDirectMemory(off_t a, off_t b, size_t len, size_t al, int t, off_t* o) { *o = 0; return 0; }
static inline int sceKernelMapDirectMemory(void** p, size_t len, int prot, int fl, off_t o, size_t al) { *p = calloc(1, len); return 0; }
typedef unsigned long OrbisPthread;
typedef struct { int x; } OrbisPthreadAttr;
static inline int32_t scePthreadCreate(OrbisPthread* t, const OrbisPthreadAttr* a, void* (*f)(void*), void* arg, const char* n) { return -1; }
static inline int32_t scePthreadJoin(OrbisPthread t, void** r) { return 0; }
typedef int32_t OrbisKernelModule;
// carga de modulos: en el PC solo se mira que el archivo exista
#include <stdio.h>
static inline uint32_t sceKernelLoadStartModule(const char* p, size_t a, const void* b, uint32_t c, void* d, void* e)
{
    FILE* f = fopen(p, "rb");
    if (!f) return 0x80020002u;
    fclose(f);
    return 0x1234;
}
static inline int32_t sceKernelDlsym(int32_t h, const char* s, void** out) { *out = (void*)0x1; return 0; }
static inline int32_t sceKernelStopUnloadModule(OrbisKernelModule h, size_t a, const void* b, uint32_t c, void* d, int32_t* e) { return 0; }
