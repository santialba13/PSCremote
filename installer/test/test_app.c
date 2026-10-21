// Test de punta a punta de la app en el PC: arma un sistema de archivos falso
// (./tfs), maneja la app con un mando guionado y revisa lo que escribio.
//   cc -DROOT='"./tfs"' -Itest/fake -Isource test/test_app.c source/inst_core.c ../plugin/source/config.c -lm
#define main app_main
#include "../source/main.c"
#undef main

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FALLO: " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

static void put(const char* path, const char* text)
{
    FILE* f = fopen(path, "wb");
    fputs(text, f);
    fclose(f);
}

static char* get(const char* path) { return read_text(path); }

static int exists(const char* path)
{
    struct stat s;
    return stat(path, &s) == 0;
}

static void write_sfo(const char* path, const char* title)
{
    unsigned char b[256] = { 0, 'P', 'S', 'F', 1, 1, 0, 0 };
    const char keys[] = "TITLE\0";
    int keyoff = 20 + 16, dataoff = keyoff + (int)sizeof keys + 1;
    b[8] = keyoff; b[12] = dataoff; b[16] = 1;
    memcpy(b + keyoff, keys, sizeof keys);
    b[20 + 4] = (unsigned char)(strlen(title) + 1); b[20 + 8] = 128;
    memcpy(b + dataoff, title, strlen(title) + 1);
    FILE* f = fopen(path, "wb"); fwrite(b, 1, sizeof b, f); fclose(f);
}

static void make_fs(void)
{
    system("rm -rf tfs && mkdir -p tfs/user/app tfs/user/appmeta tfs/data/GoldHEN/plugins tfs/data/vcfg_nx tfs/app0/plugins");
    const char* ids[] = { "CUSA00003", "CUSA00001", "CUSA00002" };
    const char* titles[] = { "Gamma", "Alpha", "beta" };
    for (int i = 0; i < 3; ++i) {
        char p[128];
        snprintf(p, sizeof p, "tfs/user/app/%s", ids[i]); mkdir(p, 0755);
        snprintf(p, sizeof p, "tfs/user/appmeta/%s", ids[i]); mkdir(p, 0755);
        snprintf(p, sizeof p, "tfs/user/appmeta/%s/param.sfo", ids[i]); write_sfo(p, titles[i]);
    }
    // un juego sin appmeta: nombre e icono salen de su app.pkg (aca, el PKG de VCFG-NX)
    mkdir("tfs/user/app/CUSA00009", 0755);
    system("cp ../../dist/VCFG-NX_v2.54.pkg tfs/user/app/CUSA00009/app.pkg");
    put("tfs/data/GoldHEN/plugins.ini", "[default]\n/data/GoldHEN/plugins/otro.prx\n/data/GoldHEN/plugins/vcfg16.prx\n\n"
        "[CUSA00003]\n" PLUGIN_LINE "\n/data/GoldHEN/plugins/gnmscale_plugin.prx\n");
    put("tfs/data/vcfg_nx/CUSA00003.ini", "sr = 1\n");
    put("tfs/app0/plugins/vcfg_nx.prx", "PRX de prueba");
    put("tfs/app0/plugins/vcfg_nx_probe.prx", "sonda de prueba");
}

static int closed_by_ps;   // el guion termina a proposito: como cerrar la app con el boton PS
void fake_pad_exhausted(void)
{
    if (!closed_by_ps) { printf("FALLO: el guion termino y la app no salio\n"); ++fails; }
    longjmp(fake_exit, 2);
}

// cada boton se aprieta y se suelta
static uint32_t script[256];
static int nscript;
static void press(uint32_t b, int times)
{
    for (int i = 0; i < times; ++i) { script[nscript++] = b; script[nscript++] = 0; }
}

static void run_app(void)
{
    fake_pad = script; fake_pad_n = nscript; fake_pad_i = 0;
    memset(games, 0, sizeof games);
    ngames = nvis = sel = top_row = filter = dirty_cfg = backup_done = jailed = 0;
    view = modal_of = V_GRID; modal_kind = M_NONE; settings_game = -1;
    free(prx); prx = NULL; prx_len = 0;
    free(probe); probe = NULL; probe_len = 0;
    memset(&diag, 0, sizeof diag);
    memset(icons, 0, sizeof icons); big_icon.game = -1;
    fake_exit_armed = 1;
    if (!setjmp(fake_exit)) app_main();
    fake_exit_armed = 0;
    fake_hide_app0 = 0;
}

int main(void)
{
    // 1. activar, ajustes propios, volver a por defecto, ajustes generales, guardar y salir
    make_fs();
    nscript = 0;
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Alpha: ON
    press(ORBIS_PAD_BUTTON_RIGHT, 1);                 // beta
    press(ORBIS_PAD_BUTTON_SQUARE, 1);                // sus ajustes
    press(ORBIS_PAD_BUTTON_DOWN, 3);                  // NXSR
    press(ORBIS_PAD_BUTTON_RIGHT, 1);                 // Normal -> Fuerte (pasa a propios)
    press(ORBIS_PAD_BUTTON_DOWN, 5);                  // Modo ahorro
    press(ORBIS_PAD_BUTTON_RIGHT, 1);                 // ON
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_RIGHT, 1);                 // Gamma (tenia ajustes propios)
    press(ORBIS_PAD_BUTTON_SQUARE, 1);
    press(ORBIS_PAD_BUTTON_SQUARE, 1);                // volver a por defecto
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_TRIANGLE, 1);              // ajustes por defecto
    press(ORBIS_PAD_BUTTON_DOWN, 3);                  // SMAA
    press(ORBIS_PAD_BUTTON_RIGHT, 1);                 // OFF
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);                // Salir...
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // ...Salir (guarda lo pendiente)
    fake_hide_app0 = 1;                               // como en la consola: fuera del sandbox no hay /app0
    run_app();
    CHECK(ngames == 4 && !strcmp(games[0].title, "Alpha") && !strcmp(games[1].title, "beta"), "lista ordenada");
    CHECK(!strcmp(games[3].title, "VCFG-NX"), "nombre desde el app.pkg: '%s'", games[3].title);
    CHECK(diag.icons_from_pkg >= 1 && diag.icons_ok >= 1, "icono desde el app.pkg (ok %d, pkg %d, fallos %d)",
          diag.icons_ok, diag.icons_from_pkg, diag.icons_fail);
    char* ini = get(PLUGINS_INI);
    CHECK(inst_ini_has(ini, "CUSA00001") && !inst_ini_has(ini, "CUSA00002") && inst_ini_has(ini, "CUSA00003"),
          "plugins.ini con Alpha y Gamma: %s", ini);
    CHECK(strstr(ini, "otro.prx") != NULL, "plugins.ini perdio otro plugin");
    CHECK(!strstr(ini, "gnmscale") && !strstr(ini, "vcfg16"), "no se quitaron los plugins que chocan: %s", ini);
    free(ini);
    CHECK(system("ls tfs/data/GoldHEN/plugins.ini.vcfg_nx_backup_* >/dev/null 2>&1") == 0, "sin copia de seguridad");
    char* cfg = get(CFG_DIR "/config.ini");
    CHECK(strstr(cfg, "\naa = 0") != NULL, "config.ini sin el cambio de SMAA: %s", cfg);
    free(cfg);
    char* g2 = get(CFG_DIR "/CUSA00002.ini");
    CHECK(strstr(g2, "\nsr = 3") && strstr(g2, "\neco = 1") && strstr(g2, "\naa = 1"),
          "ajustes propios de beta: %s", g2);
    free(g2);
    CHECK(!exists(CFG_DIR "/CUSA00003.ini"), "Gamma volvio a por defecto pero su .ini sigue");
    char* plug = get(PLUGIN_DST);
    CHECK(!strcmp(plug, "PRX de prueba"), "el plugin instalado no es el de la app (/app0 oculto tras el jailbreak)");
    free(plug);
    {
        struct stat st1, st3;
        CHECK(stat(PLUGIN_DST, &st1) == 0 && (st1.st_mode & 0777) == 0777
              && stat(PLUGINS_INI, &st3) == 0 && (st3.st_mode & 0777) == 0777,
              "permisos: el plugin, la sonda y plugins.ini tienen que quedar 0777");
    }
    CHECK(!exists(PROBE_DST), "la sonda ya no se instala");
    ini = get(PLUGINS_INI);
    CHECK(strstr(ini, "[CUSA00001]\n" PLUGIN_LINE "\n") != NULL && !strstr(ini, PROBE_LINE), "plugin (sin sonda) en la seccion: %s", ini);
    free(ini);
    CHECK(diag.save_error[0] == 0, "error al guardar: %s", diag.save_error);
    {   // lo que guardo la app lo lee igual el plugin
        nx_config_t c;
        nx_config_defaults(&c);
        char* t = get(CFG_DIR "/config.ini"); nx_config_parse(&c, t); free(t);
        t = get(CFG_DIR "/CUSA00002.ini"); nx_config_parse(&c, t); free(t);
        CHECK(c.sr == 3 && c.eco == 1 && c.aa == 1, "el plugin leeria otra cosa");
    }

    // 2. guardado automatico: se desactiva un juego, se deja el mando quieto
    //    y se cierra la app "con el boton PS" (sin pasar por Salir)
    nscript = 0;
    press(ORBIS_PAD_BUTTON_R1, 1);                    // Activos
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // desactiva el primero (Alpha)
    for (int i = 0; i < 40; ++i) script[nscript++] = 0;   // ~0,6 s sin tocar nada
    closed_by_ps = 1;
    run_app();                                        // el guion se acaba: la app "se cierra"
    CHECK(filter == F_ON, "L1/R1 no cambio el filtro");
    char* after = get(PLUGINS_INI);
    CHECK(!inst_ini_has(after, "CUSA00001") && inst_ini_has(after, "CUSA00003"), "no se guardo solo: %s", after);
    free(after);
    // y la app instala el plugin al abrirse aunque no se toque nada
    system("rm -f tfs/data/GoldHEN/plugins/vcfg_nx.prx");
    nscript = 0;
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_UP, 1);
    run_app();
    closed_by_ps = 0;
    CHECK(exists(PLUGIN_DST), "al abrir la app no se instalo el plugin");

    // 3. error al guardar: el plugin no se puede escribir -> dialogo con el motivo, sin colgarse
    system("rm -f tfs/data/GoldHEN/plugins/vcfg_nx.prx && mkdir -p tfs/data/GoldHEN/plugins/vcfg_nx.prx/x");
    nscript = 0;
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Salir -> guardar falla
    press(ORBIS_PAD_BUTTON_DOWN, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Ver diagnostico
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);                // vuelve a Opciones
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);                // cierra
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);                // Salir...
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // falla otra vez
    press(ORBIS_PAD_BUTTON_DOWN, 2);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // ...sin guardar
    run_app();
    CHECK(strstr(diag.save_error, "-31") != NULL, "el error no explica que fallo: '%s'", diag.save_error);
    system("rm -rf tfs/data/GoldHEN/plugins/vcfg_nx.prx");

    // 3b. limpiar todo y reinstalar: borra restos de versiones anteriores
    mkdir("tfs/data/vcfg_nx/logs", 0755);
    put("tfs/data/vcfg_nx/logs/CUSA00003.log", "viejo");
    put("tfs/data/vcfg_nx/CUSA00002.ini", "sr = 3\n");
    put("tfs/data/vcfg_nx/boot-CUSA00003.txt", "pending");
    put("tfs/data/GoldHEN/plugins/vcfg_nx.prx", "PRX viejo");
    put("tfs/data/GoldHEN/plugins/vcfg_nx_old.prx", "otro viejo");
    put("tfs/data/GoldHEN/plugins/otro.prx", "de otro");
    put("tfs/data/GoldHEN/plugins.ini.vcfg_nx_backup_1", "copia vieja");
    put("tfs/data/GoldHEN/plugins.ini.vcfg_tmp", "temporal");
    nscript = 0;
    press(ORBIS_PAD_BUTTON_OPTIONS, 1);
    press(ORBIS_PAD_BUTTON_DOWN, 4);                  // Limpiar todo y reinstalar
    press(ORBIS_PAD_BUTTON_CROSS, 1);
    press(ORBIS_PAD_BUTTON_UP, 1);                    // el dialogo arranca en "Cancelar"
    press(ORBIS_PAD_BUTTON_CROSS, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Aceptar
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Salir
    run_app();
    CHECK(strstr(clean_result, "Se borraron") != NULL, "limpieza: %s", clean_result);
    printf("limpieza: %s\n", clean_result);
    CHECK(!exists(CFG_DIR "/logs/CUSA00003.log") && !exists(CFG_DIR "/CUSA00002.ini") && !exists(CFG_DIR "/boot-CUSA00003.txt"),
          "quedaron restos en /data/vcfg_nx");
    CHECK(!exists(ROOT "/data/GoldHEN/plugins/vcfg_nx_old.prx") && !exists(ROOT "/data/GoldHEN/plugins.ini.vcfg_nx_backup_1") && system("ls tfs/data/GoldHEN/plugins.ini.vcfg_nx_backup_* >/dev/null 2>&1") == 0
          && !exists(ROOT "/data/GoldHEN/plugins.ini.vcfg_tmp"), "quedaron restos en /data/GoldHEN");
    CHECK(exists(ROOT "/data/GoldHEN/plugins/otro.prx"), "borro un plugin que no es de VCFG-NX");
    plug = get(PLUGIN_DST);
    CHECK(!strcmp(plug, "PRX de prueba"), "no se reinstalo el plugin nuevo: %s", plug);
    free(plug);
    ini = get(PLUGINS_INI);
    CHECK(inst_ini_has(ini, "CUSA00003") && strstr(ini, "otro.prx"), "plugins.ini tras limpiar: %s", ini);
    free(ini);
    CHECK(exists(CFG_DIR "/config.ini"), "no se escribieron los ajustes por defecto");

    // 4. desinstalar
    nscript = 0;
    press(ORBIS_PAD_BUTTON_OPTIONS, 1);
    press(ORBIS_PAD_BUTTON_DOWN, 5);                  // Desinstalar
    press(ORBIS_PAD_BUTTON_CROSS, 1);
    press(ORBIS_PAD_BUTTON_UP, 1);                    // el dialogo arranca en "Cancelar"
    press(ORBIS_PAD_BUTTON_CROSS, 1);
    run_app();
    ini = get(PLUGINS_INI);
    CHECK(!strstr(ini, PLUGIN_LINE) && strstr(ini, "otro.prx"), "desinstalar: %s", ini);
    free(ini);
    CHECK(!exists(PLUGIN_DST) && !exists(PROBE_DST), "desinstalar no borro el plugin o la sonda");
    CHECK(exists(CFG_DIR "/config.ini"), "desinstalar borro los ajustes");

    // 5. sin juegos ni GoldHEN ni plugin: no se cuelga y lo explica
    system("rm -rf tfs && mkdir -p tfs");
    nscript = 0;
    press(ORBIS_PAD_BUTTON_RIGHT, 2);
    press(ORBIS_PAD_BUTTON_SQUARE, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);
    press(ORBIS_PAD_BUTTON_L1, 1);
    press(ORBIS_PAD_BUTTON_CIRCLE, 1);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Salir -> no hay plugin en la app
    press(ORBIS_PAD_BUTTON_DOWN, 2);
    press(ORBIS_PAD_BUTTON_CROSS, 1);                 // Salir sin guardar
    run_app();
    CHECK(ngames == 0, "sin juegos");
    CHECK(strstr(diag.save_error, "-30") != NULL, "sin plugin en la app: '%s'", diag.save_error);
    system("rm -rf tfs");
    printf(fails ? "%d FALLOS\n" : "OK: app de punta a punta\n", fails);
    return fails != 0;
}
