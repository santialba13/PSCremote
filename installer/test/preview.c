// Renderiza las pantallas del instalador a PPM en el PC (stubs en test/fake/).
//   cc -Itest/fake -Isource test/preview.c source/inst_core.c -o preview && ./preview
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
int main(void)
{
    video_init();
    const char* t[] = { "Bloodborne", "Red Dead Redemption 2", "The Last of Us Remastered", "God of War",
                        "Ghost of Tsushima", "Marvel's Spider-Man", "Days Gone", "Horizon Zero Dawn" };
    const char* id[] = { "CUSA00207", "CUSA08519", "CUSA00552", "CUSA07408", "CUSA11456", "CUSA02299", "CUSA08966", "CUSA07402" };
    ngames = 8;
    for (int i = 0; i < 8; ++i) { snprintf(games[i].id, 16, "%s", id[i]); snprintf(games[i].title, 64, "%s", t[i]); games[i].enabled = i < 3; }
    snprintf(msg, sizeof msg, "Eleg\xed los juegos donde usar VCFG-NX y guard\xe1 con O.");
    sel = 1; draw_games(); dump("preview_juegos.ppm");
    draw_settings(1); dump("preview_ajustes.ppm");
    return 0;
}
