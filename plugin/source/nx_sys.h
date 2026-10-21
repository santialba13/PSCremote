// Tipos, atributos de exportacion, log y notificaciones del plugin.
// Reemplaza al plugin_common.h del repositorio de plugins de GoldHEN para que
// el plugin compile solo con OpenOrbis + GoldHEN SDK.
#pragma once
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <orbis/libkernel.h>

typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;

#define attr_public __attribute__((visibility("default")))
#define attr_module_hidden __attribute__((weak)) __attribute__((visibility("hidden")))

#define TEX_ICON_SYSTEM "cxml://psnotification/tex_icon_system"
// sin stdio: el plugin corre dentro del arranque de la libc del juego (GoldHEN
// lo carga desde _init_env) y ahi stdio puede fallar (VCFG lo documenta igual)
#define final_printf(fmt, ...) do { } while (0)

static inline void Notify(const char* icon, const char* fmt, ...)
{
    OrbisNotificationRequest req;
    memset(&req, 0, sizeof req);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(req.message, sizeof req.message, fmt, ap);
    va_end(ap);
    final_printf("notify: %s\n", req.message);
    req.type = NotificationRequest;
    req.useIconImageUri = 1;
    req.targetId = -1;
    strncpy(req.iconUri, icon, sizeof req.iconUri - 1);
    sceKernelSendNotificationRequest(0, &req, sizeof req, 0);
}
