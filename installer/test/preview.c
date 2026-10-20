// Renderiza las pantallas de la app a PPM en el PC (stubs en test/fake/).
//   cc -Itest/fake -Isource test/preview.c source/inst_core.c ../plugin/source/config.c -o preview -lm
//   ./preview carpeta_con_iconos     (carpeta/<ID>/icon0.png, opcional)
void fake_pad_exhausted(void) {}
#define main app_main
#include "../source/main.c"
#undef main
static void dump(const char* path)
{
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6 %d %d 255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        uint32_t c = fb[cur_fb][i];
        unsigned char rgb[3] = { (unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}
int main(int argc, char** argv)
{
    video_init();
    const char* t[] = { "Bloodborne", "Red Dead Redemption 2", "The Last of Us Remastered", "God of War",
                        "Ghost of Tsushima", "Marvel's Spider-Man", "Days Gone", "Horizon Zero Dawn",
                        "Uncharted 4: A Thief's End", "Gran Turismo Sport", "Persona 5 Royal", "Sekiro" };
    const char* id[] = { "CUSA00207", "CUSA08519", "CUSA00552", "CUSA07408", "CUSA11456", "CUSA02299",
                         "CUSA08966", "CUSA07402", "CUSA00341", "CUSA03220", "CUSA17416", "CUSA12408" };
    nx_config_defaults(&defaults);
    ngames = 12;
    for (int i = 0; i < ngames; ++i) {
        snprintf(games[i].id, 16, "%s", id[i]);
        snprintf(games[i].title, 64, "%s", t[i]);
        if (argc > 1) snprintf(games[i].meta, sizeof games[i].meta, "%s/%s", argv[1], id[i]);
        games[i].enabled = i % 3 != 2;
        games[i].cfg = defaults;
    }
    games[1].custom = 1; games[1].cfg.sr = 3; games[1].cfg.eco = 1; games[1].cfg.temp_limit = 75;
    games[4].custom = 1; games[4].cfg.framegen = 0;
    rebuild_visible();
    set_msg(C_DIM, "Eleg\xed un juego: X lo activa, CUADRADO abre sus ajustes. O guarda y sale.");
    sel = 1; draw_grid(); dump("ui_juegos.ppm");
    draw_settings(1, R_SR); dump("ui_juego_propio.ppm");
    draw_settings(0, R_MODE); dump("ui_juego_defecto.ppm");
    draw_settings(-1, R_ECO); dump("ui_defecto.ppm");
    draw_options(3); dump("ui_opciones.ppm");
    draw_confirm(O_UNINSTALL); dump("ui_confirmar.ppm");
    filter = F_CUSTOM; rebuild_visible(); draw_grid(); dump("ui_filtro.ppm");
    // 30 juegos: desplazamiento
    for (int i = 12; i < 30; ++i) {
        games[i] = games[i % 12];
        snprintf(games[i].id, 16, "CUSA9%04d", i);
    }
    ngames = 30; filter = F_ALL; rebuild_visible(); sel = 22; draw_grid(); dump("ui_scroll.ppm");
    return 0;
}
