// Renderiza las pantallas de la app a PPM en el PC (stubs en test/fake/), comprueba
// que el redibujo parcial deja exactamente lo mismo que un redibujo completo y
// mide los tiempos de dibujo.
//   make -C installer/test preview ICONS=carpeta     (carpeta/<ID>/icon0.png, opcional)
#include <time.h>
void fake_pad_exhausted(void) {}
#define main app_main
#include "../source/main.c"
#undef main

static void dump(const char* path)
{
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6 %d %d 255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        uint32_t c = canvas.px[i];
        unsigned char rgb[3] = { (unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int fails;

// el lienzo despues de redibujos parciales tiene que ser igual a un redibujo completo
static void check_partial(const char* what)
{
    size_t n = (size_t)W * H * 4;
    uint32_t* a = malloc(n);
    memcpy(a, canvas.px, n);
    repaint_all();
    int diff = 0;
    for (size_t i = 0; i < (size_t)W * H; ++i) diff += a[i] != canvas.px[i];
    if (diff) { printf("FALLO: redibujo parcial distinto del completo (%s): %d pixeles\n", what, diff); ++fails; }
    free(a);
}

int main(int argc, char** argv)
{
    video_init();
    build_background();
    const char* t[] = { "Bloodborne", "Red Dead Redemption 2", "The Last of Us Remastered", "God of War",
                        "Ghost of Tsushima", "Marvel's Spider-Man", "Days Gone", "Horizon Zero Dawn",
                        "Uncharted 4: A Thief's End", "Gran Turismo Sport", "Persona 5 Royal", "Sekiro: Shadows Die Twice" };
    const char* id[] = { "CUSA00207", "CUSA08519", "CUSA00552", "CUSA07408", "CUSA11456", "CUSA02299",
                         "CUSA08966", "CUSA07402", "CUSA00341", "CUSA03220", "CUSA17416", "CUSA12408" };
    nx_config_defaults(&defaults);
    ngames = 12;
    for (int i = 0; i < ngames; ++i) {
        snprintf(games[i].id, 16, "%s", id[i]);
        snprintf(games[i].title, 64, "%s", t[i]);
        if (argc > 1) snprintf(games[i].meta, sizeof games[i].meta, "%s/%s", argv[1], id[i]);
        // argv[2]: carpeta con <ID>/app.pkg (PKG reales): esos juegos toman el icono del PKG
        if (argc > 2 && i < 3) {
            games[i].meta[0] = 0;
            snprintf(games[i].app, sizeof games[i].app, "%s/%s", argv[2], id[i]);
        }
        games[i].enabled = i % 3 != 2;
        games[i].cfg = defaults;
    }
    games[1].custom = 1; games[1].cfg.sr = 3; games[1].cfg.eco = 1; games[1].cfg.temp_limit = 75;
    games[4].custom = 1; games[4].cfg.framegen = 0;
    prx = (uint8_t*)"x"; prx_len = 1;
    snprintf(prx_from, sizeof prx_from, "/app0/plugins/vcfg_nx.prx");
    diag.sdk = 0x0204;
    rebuild_visible();
    build_sprites();
    set_msg(C_DIM, "Eleg\xed un juego con las flechas.");
    sel = 1;
    repaint_all();
    while (load_one_icon()) {}
    dump("ui_juegos.ppm");

    // tiempos: redibujo completo y movimiento entre tarjetas
    double t0 = now_ms();
    for (int i = 0; i < 10; ++i) repaint_all();
    double full = (now_ms() - t0) / 10;
    t0 = now_ms();
    for (int i = 0; i < 40; ++i) grid_move(i % 2 ? 1 : 2);
    double move = (now_ms() - t0) / 40;
    check_partial("mover en la cuadricula");
    // cambio de pagina con 30 juegos
    for (int i = 12; i < 30; ++i) { games[i] = games[i % 12]; snprintf(games[i].id, 16, "CUSA9%04d", i); }
    ngames = 30; rebuild_visible();
    repaint_all();
    for (int k = 2; k < 25; k += 6) grid_move(k);
    while (load_one_icon()) {}
    check_partial("cambio de pagina");
    dump("ui_scroll.ppm");
    ngames = 12; rebuild_visible(); sel = 1; top_row = 0;

    view = V_GAME; settings_game = 1; settings_row = R_SR;
    repaint_all();
    while (load_one_icon()) {}
    dump("ui_juego_propio.ppm");
    t0 = now_ms();
    for (int i = 0; i < 20; ++i) handle_settings(i % 2 ? ORBIS_PAD_BUTTON_UP : ORBIS_PAD_BUTTON_DOWN);
    double row = (now_ms() - t0) / 20;
    check_partial("filas de ajustes");
    settings_game = 0; settings_row = R_MODE; big_icon.game = -2;
    repaint_all();
    while (load_one_icon()) {}
    dump("ui_juego_defecto.ppm");
    settings_game = -1; settings_row = R_ECO; repaint_all(); dump("ui_defecto.ppm");

    view = V_GRID; repaint_all();
    open_modal(M_OPTIONS); modal.row = 4; repaint_modal_options(); check_partial("opciones");
    dump("ui_opciones.ppm");
    open_modal(M_EXIT); dump("ui_salir.ppm");
    snprintf(diag.save_error, sizeof diag.save_error, "%s (c\xf3" "digo -31)", save_error_text(-31));
    open_modal(M_SAVE_ERROR); dump("ui_error.ppm");
    open_modal(M_DIAG); dump("ui_diagnostico.ppm");
    open_modal(M_SAVED); dump("ui_guardado.ppm");
    close_modal();
    filter = F_CUSTOM; rebuild_visible(); repaint_all(); dump("ui_filtro.ppm");

    printf("dibujo en este PC: pantalla completa %.1f ms, mover entre juegos %.1f ms, mover en ajustes %.1f ms\n",
           full, move, row);
    printf(fails ? "%d FALLOS\n" : "OK: redibujo parcial = completo\n", fails);
    return fails != 0;
}
