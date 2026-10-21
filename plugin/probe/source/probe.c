// Sonda de VCFG-NX: un plugin minimo (solo libkernel) que la app pone ANTES de
// vcfg_nx.prx en la seccion del juego. Sirve para saber, sin buscar archivos,
// en que paso se corta:
//   * si no aparece ni su aviso, GoldHEN no cargo plugins en este juego
//     (plugins.ini, version de GoldHEN, plugins desactivados);
//   * si aparece pero dice que el plugin principal no esta, GoldHEN no pudo
//     abrir vcfg_nx.prx en este juego.
#include <Common.h>
#include <orbis/libkernel.h>
#include <stdio.h>
#include <string.h>

#include "../../source/nx_sys.h"

attr_public const char* g_pluginName = "vcfg_nx_probe";
attr_public const char* g_pluginDesc = "Sonda de diagnostico de VCFG-NX";
attr_public const char* g_pluginAuth = "VCFG-NX";
attr_public u32 g_pluginVersion = 0x00000265; // 2.65

static char g_title[16];
static OrbisPthread g_thread;

static void append(const char* line)
{
    sceKernelMkdir("/data/vcfg_nx", 0777);
    sceKernelMkdir("/data/vcfg_nx/logs", 0777);
    int fd = sceKernelOpen("/data/vcfg_nx/logs/cargas.txt", 0x0001 | 0x0200 | 0x0008, 0777);
    if (fd < 0) return;
    sceKernelWrite(fd, line, strlen(line));
    sceKernelClose(fd);
}

// 1 si el modulo del plugin principal esta cargado en el proceso
static int main_loaded(void)
{
    OrbisKernelModule mods[256];
    size_t n = 0;
    if (sceKernelGetModuleList(mods, 256, &n) != 0) return -1;
    if (n > 256) n = 256;
    for (size_t i = 0; i < n; ++i) {
        OrbisKernelModuleInfo info;
        memset(&info, 0, sizeof info);
        info.size = sizeof info;
        if (sceKernelGetModuleInfo(mods[i], &info) != 0) continue;
        if (!strcmp(info.name, "vcfg_nx") || !strcmp(info.name, "vcfg_nx.prx")) return 1;
    }
    return 0;
}

static void* check_later(void* arg)
{
    (void)arg;
    sceKernelUsleep(12 * 1000000);
    int m = main_loaded();
    char msg[200];
    snprintf(msg, sizeof msg, m > 0 ? "VCFG-NX sonda: el plugin principal esta cargado en %s"
                                    : "VCFG-NX sonda: GoldHEN NO pudo cargar vcfg_nx.prx en %s",
             g_title);
    Notify(TEX_ICON_SYSTEM, "%s", msg);
    char line[240];
    snprintf(line, sizeof line, "[sonda] %s (modulo principal: %d)\n", msg, m);
    append(line);
    return NULL;
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    struct proc_info info;
    if (sys_sdk_proc_info(&info) == 0) memcpy(g_title, info.titleid, sizeof g_title - 1);
    char line[160];
    snprintf(line, sizeof line, "[sonda] GoldHEN cargo plugins en %s\n", g_title);
    append(line);
    Notify(TEX_ICON_SYSTEM, "VCFG-NX sonda: GoldHEN carg\xc3\xb3 plugins en %s", g_title);
    scePthreadCreate(&g_thread, NULL, check_later, NULL, "vcfg_nx_probe");
    return 0;
}

s32 attr_public plugin_unload(s32 argc, const char* argv[]) { return 0; }
s32 attr_module_hidden module_start(s64 argc, const void* args) { return 0; }
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
