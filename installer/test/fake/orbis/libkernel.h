#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
typedef struct { int type, reqId, priority, msgId, targetId, userId, unk1, unk2, appId, errorNum, unk3; char useIconImageUri; char message[1024]; char iconUri[1024]; char unk[1024]; } OrbisNotificationRequest;
enum { NotificationRequest = 0 };
static inline int sceKernelSendNotificationRequest(int a, OrbisNotificationRequest* r, size_t s, int b) { return 0; }
static inline int sceKernelUsleep(unsigned u) { return 0; }
static inline size_t sceKernelGetDirectMemorySize(void) { return 1u << 30; }
static inline int sceKernelAllocateDirectMemory(off_t a, off_t b, size_t len, size_t al, int t, off_t* o) { *o = 0; return 0; }
static inline int sceKernelMapDirectMemory(void** p, size_t len, int prot, int fl, off_t o, size_t al) { *p = calloc(1, len); return 0; }
