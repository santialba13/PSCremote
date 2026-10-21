// Test en PC de la logica del instalador: plugins.ini y param.sfo.
//   cc -I../source test_core.c ../source/inst_core.c -o test_core && ./test_core
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "inst_core.h"

static int fails;
#define CHECK(c, m) do { if (!(c)) { printf("FALLO: %s\n", m); ++fails; } } while (0)

static void write_sfo(const char* path, const char* title)
{
    // param.sfo minimo con dos claves (APP_VER, TITLE)
    unsigned char b[256] = { 0, 'P', 'S', 'F', 1, 1, 0, 0 };
    const char keys[] = "APP_VER\0TITLE\0";
    int nk = 2, keyoff = 20 + 16 * nk, dataoff = keyoff + (int)sizeof keys + 1;
    b[8] = keyoff; b[12] = dataoff; b[16] = nk;
    memcpy(b + keyoff, keys, sizeof keys);
    unsigned char* e = b + 20;
    e[0] = 0; e[4] = 6; e[8] = 8; e[12] = 0;                    // APP_VER
    memcpy(b + dataoff, "01.00", 6);
    e += 16; e[0] = 8; e[4] = (unsigned char)(strlen(title) + 1); e[8] = 128; e[12] = 8;   // TITLE
    memcpy(b + dataoff + 8, title, strlen(title) + 1);
    FILE* f = fopen(path, "wb"); fwrite(b, 1, sizeof b, f); fclose(f);
}

int main(void)
{
    const char* ini0 = "; plugins de GoldHEN\n[default]\n/data/GoldHEN/plugins/afr.prx\n\n[CUSA00207]\n/data/GoldHEN/plugins/x.prx\n";
    char* a = inst_ini_add(ini0, "CUSA00207");
    CHECK(a && inst_ini_has(a, "CUSA00207"), "agregar en seccion existente");
    CHECK(strstr(a, "x.prx") && strstr(a, "afr.prx"), "no conservo lo existente");
    CHECK(inst_ini_add(a, "CUSA00207") == NULL, "no es idempotente");
    char* b = inst_ini_add(a, "CUSA08519");
    CHECK(b && inst_ini_has(b, "CUSA08519") && !inst_ini_has(b, "default"), "seccion nueva");
    char* c = inst_ini_remove(b, "CUSA00207");
    CHECK(c && !inst_ini_has(c, "CUSA00207") && inst_ini_has(c, "CUSA08519") && strstr(c, "x.prx"), "quitar de una seccion");
    char* d = inst_ini_remove_all(c);
    CHECK(d && !strstr(d, "vcfg_nx.prx") && !strstr(d, "vcfg_nx_probe") && strstr(d, "afr.prx") && strstr(d, "x.prx"), "desinstalar");
    printf("--- plugins.ini tras agregar 2 juegos ---\n%s---\n", b);

    mkdir("tmp_app", 0755); mkdir("tmp_app/CUSA00207", 0755); mkdir("tmp_app/CUSA08519", 0755);
    mkdir("tmp_app/NPXS00001", 0755);
    mkdir("tmp_meta", 0755); mkdir("tmp_meta/CUSA00207", 0755); mkdir("tmp_meta/CUSA08519", 0755);
    mkdir("tmp_meta/external", 0755); mkdir("tmp_meta/external/CUSA07408", 0755);
    mkdir("tmp_ext", 0755); mkdir("tmp_ext/CUSA07408", 0755); mkdir("tmp_ext/CUSA00207", 0755);
    write_sfo("tmp_meta/CUSA00207/param.sfo", "Bloodborne");
    write_sfo("tmp_meta/CUSA08519/param.sfo", "Red Dead Redemption 2");
    write_sfo("tmp_meta/external/CUSA07408/param.sfo", "god of War");
    const char* metas[] = { "tmp_meta", "tmp_meta/external", NULL };
    inst_game_t g[8];
    int n = inst_list_games("tmp_app", metas, g, 0, 8);
    CHECK(n == 2, "lista de juegos (deberia ignorar NPXS)");
    CHECK(n == 2 && !strcmp(g[0].title, "Bloodborne") && !strcmp(g[1].title, "Red Dead Redemption 2"), "titulos del param.sfo");
    n = inst_list_games("tmp_ext", metas, g, n, 8);   // disco externo: CUSA00207 repetido
    CHECK(n == 3, "juegos del disco externo sin repetir");
    CHECK(n == 3 && !strcmp(g[1].title, "god of War") && !strcmp(g[1].meta, "tmp_meta/external/CUSA07408"),
          "orden sin mayusculas y meta externa");
    for (int i = 0; i < n; ++i) printf("juego: %s  %s  (%s)\n", g[i].id, g[i].title, g[i].meta);

    // ajustes por juego
    nx_config_t def;
    nx_config_defaults(&def);
    def.aa = 0;
    mkdir("tmp_cfg", 0755);
    FILE* f = fopen("tmp_cfg/CUSA08519.ini", "w");
    fputs("sr = 3\neco = 1\ntemp_limit = 75\n", f);
    fclose(f);
    inst_load_game_cfg(&g[0], &def, "tmp_cfg");
    CHECK(!g[0].custom && g[0].cfg.aa == 0, "sin .ini: ajustes por defecto");
    inst_load_game_cfg(&g[2], &def, "tmp_cfg");
    CHECK(g[2].custom && g[2].cfg.sr == 3 && g[2].cfg.eco && g[2].cfg.temp_limit == 75 && g[2].cfg.aa == 0,
          "con .ini: por defecto + lo propio");
    // entradas de un PKG real (el de VCFG-NX): icon0.png y param.sfo sin cifrar
    size_t ilen = 0, slen = 0;
    unsigned char* icon = inst_pkg_entry("../../dist/VCFG-NX_v2.53.pkg", PKG_ENTRY_ICON0, &ilen);
    CHECK(icon && ilen > 1000 && !memcmp(icon, "\x89PNG", 4), "icon0.png del PKG");
    unsigned char* sfo = inst_pkg_entry("../../dist/VCFG-NX_v2.53.pkg", PKG_ENTRY_PARAM_SFO, &slen);
    char ptitle[64] = "";
    CHECK(sfo && inst_sfo_title(sfo, slen, ptitle, sizeof ptitle) == 0 && !strcmp(ptitle, "VCFG-NX"), "param.sfo del PKG");
    free(icon); free(sfo);
    CHECK(inst_pkg_entry("test_core.c", PKG_ENTRY_ICON0, &ilen) == NULL, "un archivo que no es PKG");
    CHECK(inst_pkg_entry("no_existe.pkg", PKG_ENTRY_ICON0, &ilen) == NULL, "PKG inexistente");
    char txt[512];
    nx_config_format(&g[2].cfg, "test", txt, sizeof txt);
    nx_config_t back;
    nx_config_defaults(&back);
    nx_config_parse(&back, txt);
    CHECK(!memcmp(&back.framegen, &g[2].cfg.framegen, sizeof(int)) && back.sr == 3 && back.eco == 1 && back.aa == 0
          && back.temp_limit == 75 && back.dynamic == g[2].cfg.dynamic, "formato del .ini ida y vuelta");
    char oth[200];
    const char* ini2 = "[default]\r\n/data/GoldHEN/plugins/afr.prx\r\n[CUSA00411]\r\n/data/GoldHEN/plugins/gnmscale_plugin.prx\r\n"
                       "  /data/GoldHEN/plugins/vcfg_nx.prx\r\n;/data/GoldHEN/plugins/x.prx\r\n[CUSA1]\n";
    CHECK(inst_ini_section_plugins(ini2, "CUSA00411", oth, sizeof oth) == 1 && !strcmp(oth, "gnmscale_plugin.prx"),
          "plugins de una seccion (CRLF, comentarios, otros plugins)");
    CHECK(inst_ini_section_plugins(ini2, "CUSA1", oth, sizeof oth) == 0 && !oth[0], "seccion sin plugins");
    {   // conflictos: GNMScale, VCFG viejos y plugins de framerate salen; los demas quedan
        const char* ini3 = "[default]\n/data/GoldHEN/plugins/vcfg16.prx\n/data/GoldHEN/plugins/game_patch.prx\n"
                           "[CUSA00411]\n/data/GoldHEN/plugins/vcfg_nx_probe.prx\n/data/GoldHEN/plugins/vcfg_nx.prx\n"
                           "/data/GoldHEN/plugins/gnmscale_plugin.prx\r\n/data/GoldHEN/plugins/afr.prx\n"
                           "/data/GoldHEN/plugins/vcfg_classic42.prx\n[CUSA00880]\n/data/GoldHEN/plugins/gnmscale_plugin.prx\n";
        char rm[200] = "";
        char* a3 = inst_ini_strip_conflicts(ini3, "CUSA00411", rm, sizeof rm);
        CHECK(a3 && !strcmp(rm, "gnmscale_plugin.prx, vcfg_classic42.prx"), "conflictos quitados");
        CHECK(a3 && strstr(a3, "vcfg_nx.prx") && strstr(a3, "vcfg_nx_probe.prx") && strstr(a3, "afr.prx")
              && strstr(a3, "[CUSA00880]\n/data/GoldHEN/plugins/gnmscale_plugin.prx"), "quito de mas");
        char* b3 = a3 ? inst_ini_strip_conflicts(a3, "default", rm, sizeof rm) : NULL;
        CHECK(b3 && !strstr(b3, "vcfg16") && strstr(b3, "game_patch.prx"), "[default]");
        CHECK(inst_ini_strip_conflicts(b3, "CUSA00411", rm, sizeof rm) == NULL, "sin conflictos deberia ser NULL");
        printf("conflictos quitados: %s\n", rm);
        free(a3); free(b3);
    }
    char sum[200];
    CHECK(inst_log_summary(NULL, sum, sizeof sum) == 0, "registro: sin archivo");
    CHECK(inst_log_summary("[      0] [engine] plugin_load\n[      0] [vo] formato 0xc1060000 1920x1080 no soportado: x\n",
                           sum, sizeof sum) == 1 && strstr(sum, "0xc1060000"), "registro: formato");
    printf("registro (formato): %s\n", sum);
    CHECK(inst_log_summary("[      0] [engine] plugin_load\n[      2] [flip] estado=1 handle=0\n[     30] [diag] 900 flips pero ningun RegisterBuffers visto\n",
                           sum, sizeof sum) == 1 && strstr(sum, "tarde"), "registro: registro antes de cargar");
    CHECK(inst_log_summary("[      0] [vo] buffers del juego: 1920x1080\n[   3600] [flip] estado=3 handle=0 idx=1 count=1 gnm=3600 wl=0 cpu=0 nuestros=2900 fps=30.0\n",
                           sum, sizeof sum) == 3 && strstr(sum, "2900 de 3600"), "registro: activo");
    printf("registro (activo): %s\n", sum);
    free(a); free(b); free(c); free(d);
    system("rm -rf tmp_app tmp_meta tmp_ext tmp_cfg");
    printf(fails ? "%d FALLOS\n" : "OK\n", fails);
    return fails != 0;
}
