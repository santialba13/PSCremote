# Análisis de GNMScale (PKG de la comunidad)

Archivo: `IV0000-GNMS00001_00-GNMSCALE00000000.pkg` (6,3 MB). Extraído con
`tools/pkg/` igual que VCFG (fPKG, passcode por defecto). No se ejecutó en
consola: todo sale de los archivos y binarios.

## Contenido

| Archivo | Qué es |
|---|---|
| `eboot.bin` | Lanzador con interfaz: lista juegos (`/user/app` y `/mnt/ext1/user/app`), configura por juego, escribe `plugins.ini` |
| `Media/jb.prx` | Sale del sandbox de la app (montajes `nullfs`) para poder leer/escribir fuera |
| `gnmscale_plugin.prx` | Plugin de GoldHEN que aplica la "super resolución" en el flip del juego |
| `models/fbsr3-pico-*-femto/fp16/` | 3 redes: `fidelity`, `sharp-fast`, `sharp` (manifest + shader `.sb` + `weights.bin`) |
| `models/fsr1-rcas/` | RCAS de AMD FSR1 como shader |
| `models/overlay/overlay.sb` | Panel del menú en pantalla |

Los textos internos están en italiano. Hay menú en juego con L3+R3, ajustes
por juego en `/data/GnmScale/`, modo "comparar mitad original", tope de FPS y
varias opciones de memoria (`mem_garlic_mb`, `mem_onion_mb`, "salida directa
a pantalla").

## Las redes

Según los `manifest.json`, las tres son la misma arquitectura `SRNetFemto`:

| Modelo | Capas | Ancho | MAC/píxel | Pesos |
|---|---|---|---|---|
| fidelity | cabeza 3x3 + cola 1x1 | 4 canales | 156 | 168 floats |
| sharp-fast | cabeza + 1 capa 3x3 + cola | 4 | 444 | 316 |
| sharp | cabeza + 2 capas 3x3 + cola | 4 | 444 | 464 |

* **Escala 1x**: no reescalan. Toman el fotograma ya hecho por el juego y le
  suman un residuo aprendido (`conv_fused_residual`), o sea, son filtros de
  nitidez aprendidos.
* La salida pasa por un factor de **intensidad** que el menú deja subir por
  encima de 1 (la app lo llama "estrapolazione: aloni e tosature attesi").
* Están entrenadas con DirectX 11 (`cs_5_0`) y portadas a PS4.

## Por qué agregan grano

`sr/eval_sr.py` carga sus `weights.bin` con el layout del manifest
(`net_sr.load_gnmscale`) y las evalúa sobre el mismo conjunto de validación que
NXSR: imágenes de Urban100, BSD100, Set14 y Big Buck Bunny, con degradaciones
típicas de un juego (render a menor resolución, desenfoque de TAA, grano).

| método | PSNR | PSNR en zonas planas | píxeles con halo |
|---|---|---|---|
| imagen sin procesar | 27.28 | 42.57 | 1.9 % |
| GNMScale fidelity | 25.16 | 35.56 | 25.6 % |
| GNMScale sharp-fast | 24.92 | 35.71 | 26.2 % |
| GNMScale sharp | 25.11 | 35.52 | 26.5 % |
| **NXSR (VCFG-NX)** | **27.68** | **42.73** | **2.2 %** |

Las tres **empeoran** la imagen respecto de no hacer nada:

* en zonas planas el PSNR cae ~7 dB: amplifican ruido y dithering, que es el
  granulado que se ve;
* uno de cada cuatro píxeles termina fuera del rango local del original
  (halos y sobreimpulsos alrededor de bordes).

Es lo esperable en un filtro de nitidez sin límites: realza todo lo que tiene
alta frecuencia, sea detalle o ruido. Con 4 canales la red no tiene capacidad
para distinguirlos.

**Salvedad:** la reimplementación supone `lrelu` con pendiente 0,1 (el manifest
no la dice) e intensidad 1. Si la pendiente real fuera otra, los números
cambiarían algo, pero el comportamiento (residuo sin acotar) es el mismo.

## Qué se toma y qué no para VCFG-NX

| GNMScale | VCFG-NX |
|---|---|
| filtro de nitidez aprendido, sin límites | NXSR con **limitador**: la salida no puede salir del rango de los 3x3 vecinos → no hay halos ni grano por construcción |
| intensidad > 1 (extrapola) | intensidad 0 a 1, nunca extrapola |
| 4 canales, 444 MAC/px | 16 canales a media resolución, ~1056 MAC/px, todo en un pase con memoria local (sin buffers intermedios) |
| entrenada en otros datos | entrenada con degradaciones de juego, y con ruido en la entrada y objetivo limpio |
| `jb.prx` propio para salir del sandbox | `sys_sdk_jailbreak` del SDK de GoldHEN, restaurando el sandbox al salir |
| lista `/user/app` y `/mnt/ext1/user/app` | igual |
| menú L3+R3 y comparar mitad original | igual, con la interfaz de VCFG |

## Versión nueva ("nofg", 01.00, git e6dfd37, octubre 2026)

Solo se estudió cómo funciona; ningún modelo suyo se usa en VCFG-NX.

**Modelos.** Además de los SRNetFemto de antes trae FBSR-v2 y FBSR-v3: "grilla de
celdas 8x8 (estadísticas + MLP 32->64->10: contexto y luz), conv 3x3 3->6, MLP por
píxel 58->12/16 (Winograd F(2,3)), cabezal polinómico sub-píxel; salida devuelta a
la resolución del fotograma (promedio por área)". `fbsr_proc` 1..4: la red corre a
1/N del lado y reconstruye. FBSR-v3 agrega una base lineal 5x5 por canal (mezcla
de 8 filtros) + residuos. Opción de "textura generada" (0 = reconstrucción fiel,
1 = textura inventada; avisan que en movimiento tiembla).

**Integración** (textos del plugin, en italiano):
- hooks de `sceGnmSubmitAndFlipCommandBuffers(ForWorkload)`, `sceGnmSubmitCommandBuffers(ForWorkload)`,
  `sceVideoOutSubmitFlip`, `RegisterBuffers`, `SetFlipRate`, `GetFlipStatus`;
- "SUL POSTO": su trabajo al final de los comandos del juego; "N buffer di uscita,
  non registrato sul video out (il risultato viene copiato nel buffer del gioco)",
  o un modo "Flip" que presenta su buffer sin copia. Es lo mismo que hace VCFG-NX;
- "il fotogramma non e' SDR a 8 bit: effetti spenti" (como nosotros con HDR); asume
  B8G8R8A8_SRGB con tiling 2D cuando tiene que recuperar la geometría;
- lee los DCB del juego (`CB_COLOR0_BASE`, profundidad, vectores de movimiento) para
  aplicar la super resolución **antes del HUD** ("Super resolution prima dell'HUD"),
  "HUDFix" y la generación sobre la escena sin HUD. Potente pero frágil: depende de
  adivinar qué buffer es cuál en cada juego (tiene una "vista de buffers" para que
  el usuario lo elija).

**Qué se tomó.** La confirmación de que la forma de integrar es la correcta, la
idea de la guía por celdas (NXSR-L) y la lección de que a 1080p nativos el
realce tiene que ser fuerte para notarse. No se copia: la extrapolación sin límite
(halos), la textura inventada (parpadeo) ni leer los DCB del juego para encontrar
buffers.
