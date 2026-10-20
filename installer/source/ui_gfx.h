// Dibujo 2D por CPU para la app: rectangulos redondeados con suavizado,
// degradados, sombras/brillos (como sprites calculados una vez), mezcla con
// transparencia y texto Inter pre-renderizado. Todo respeta un rectangulo de
// recorte, asi la app redibuja solo la zona que cambio.
// Colores: 0xAARRGGBB (alfa no premultiplicado).
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t* px; int w, h, stride; } ui_surf_t;
// sp (opcional, ui_img_spans): por fila, el primer y el ultimo tramo con alfa > 0 y
// el tramo opaco mas largo, para que ui_blit salte lo transparente
typedef struct { uint32_t* px; int w, h; int16_t* sp; } ui_img_t;

void ui_target(ui_surf_t* s);
void ui_clip(int x, int y, int w, int h);   // interseccion con la superficie
void ui_clip_all(void);

void ui_fill(int x, int y, int w, int h, uint32_t c);   // mezcla si alfa < 255
void ui_rrect(int x, int y, int w, int h, int r, uint32_t c);
void ui_rrect_grad(int x, int y, int w, int h, int r, uint32_t top, uint32_t bottom);
void ui_rrect_stroke(int x, int y, int w, int h, int r, int t, uint32_t c);
void ui_rrect_stroke_grad(int x, int y, int w, int h, int r, int t, uint32_t top, uint32_t bottom);
// vidrio: copia los pixeles de src (del mismo tamano que el destino) dentro del
// rectangulo redondeado, con bordes suavizados
void ui_rrect_src(int x, int y, int w, int h, int r, const ui_surf_t* src);
// desenfoque (~gaussiano, 3 pasadas de caja) de una zona de s, en el lugar
void ui_blur(ui_surf_t* s, int x, int y, int w, int h, int radius);
void ui_circle(int cx, int cy, int r, uint32_t c);
// copia una zona de otra superficie del mismo tamano (restaurar el fondo)
void ui_copy(const ui_surf_t* src, int x, int y, int w, int h);

// sprites (malloc; liberar con ui_img_free)
ui_img_t ui_img_new(int w, int h);
void ui_img_free(ui_img_t* im);
// sombra/brillo difuso de un rectangulo redondeado w x h: el sprite mide
// (w + 2*blur) x (h + 2*blur); dibujarlo en (x - blur, y - blur)
ui_img_t ui_make_shadow(int w, int h, int r, int blur, uint32_t c);
// recorta las esquinas de una imagen (alfa) con suavizado
void ui_img_round(ui_img_t* im, int r);
// borra (alfa 0) el interior de un rectangulo redondeado: la parte de una sombra
// que siempre queda tapada por la tarjeta
void ui_img_hole(ui_img_t* im, int x, int y, int w, int h, int r);
void ui_img_spans(ui_img_t* im);
void ui_blit(const ui_img_t* im, int x, int y, int alpha);   // alpha global 0..255

// texto: y es la parte de arriba de la linea
int ui_text(int font, int x, int y, uint32_t c, const char* s);
int ui_text_w(int font, const char* s);
int ui_line_h(int font);
// recorta con "..." para que entre en maxw
void ui_text_fit(int font, const char* s, int maxw, char* out, size_t n);
// parte en dos lineas por palabras (la segunda con "..." si no entra)
void ui_text_wrap2(int font, const char* s, int maxw, char* l1, char* l2, size_t n);

// iconos de los botones del mando
enum { UI_BTN_CROSS, UI_BTN_CIRCLE, UI_BTN_SQUARE, UI_BTN_TRIANGLE };
void ui_button(int which, int x, int y);
int ui_button_size(void);
