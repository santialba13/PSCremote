# Resultados

Todo medido en PC (CPU). Ningún número de esta página viene de una consola.

## 1. Generación de fotogramas

Se predice el fotograma `t = ½` a partir de `t = 0` y `t = 1`, y se compara con
el real. "block-matching (tipo VCFG)" es `model/baseline_bm.py`, que reimplementa
la parte central del original (pirámide + búsqueda por bloques + mediana + warp).

| conjunto | método | PSNR (dB) | SSIM |
|---|---|---|---|
| video (recortes 192 px, Big Buck Bunny, fotogramas no vistos) | repetir fotograma | 33.45 | 0.9543 |
| | mezcla 50/50 | 36.61 | 0.9653 |
| | block-matching (tipo VCFG) | 37.95 | 0.9754 |
| | **NXFlow híbrido (nuevo)** | **37.95** | **0.9759** |
| sintético tipo juego (HUD, oclusiones, giros de cámara) | repetir fotograma | 19.34 | 0.6044 |
| | mezcla 50/50 | 21.76 | 0.6463 |
| | block-matching (tipo VCFG) | 24.44 | 0.7906 |
| | **NXFlow híbrido (nuevo)** | **24.60** | **0.7960** |
| video completo 1280x720 | repetir fotograma | 34.45 | 0.9824 |
| | mezcla 50/50 | 34.51 | 0.9846 |
| | block-matching (tipo VCFG) | 37.03 | 0.9890 |
| | **NXFlow híbrido (nuevo)** | **37.07** | **0.9893** |

### Lectura honesta

* El enfoque de VCFG ya es bueno: +1,3 a +2,7 dB sobre la mezcla simple.
* El modelo nuevo **mejora poco** sobre él: +0,16 dB y +0,005 SSIM en el
  sintético, y prácticamente empate en video. Nunca rinde peor (por diseño
  arranca siendo el block-matching).
* Un análisis oráculo (elegir en cada píxel el mejor de los dos warps
  mirando el fotograma real) indica que hay ~+2 dB disponibles solo en manejo
  de oclusiones. La red, entrenada ~20 000 pasos en CPU con ~5 400 tripletes,
  captura una fracción mínima: la máscara aprendida casi no se aparta de 0,5.
* Experimentos que no mejoraron: red pura sin block-matching (se quedó al
  nivel de la mezcla 50/50), nivel extra a 1/4 sin supervisión (igual), peso
  de supervisión de máscara 15x mayor (−0,1 dB en video).

Para llegar a esos +2 dB hace falta más entrenamiento del que se puede hacer
en CPU: más datos (Vimeo-90k, y sobre todo **capturas del juego objetivo**) y
entrenar en GPU. Toda la tubería (datos → entrenamiento → export → shaders →
plugin) ya está lista para reentrenar sin tocar el código de la consola,
siempre que la arquitectura se mantenga.

## 2. Escalado (FSR1)

Big Buck Bunny 1280x720, reducido a 960x540 y vuelto a escalar:

| método | PSNR (dB) |
|---|---|
| bilineal | 39.42 |
| bicúbico | 41.48 |
| Lanczos | 41.77 |
| EASU (FSR1) | 40.58 |
| EASU + RCAS | 36.74 |

Como se esperaba, FSR1 no gana en PSNR: EASU está pensado para bordes nítidos
sin *ringing* y RCAS agrega nitidez percibida, que el PSNR penaliza. Frente al
escalador bilineal de VideoOut (lo que usa la consola cuando un juego entrega
900p/720p) EASU sí es mejor (+1,2 dB). La ventaja real de esta etapa es
perceptual y conviene juzgarla mirando (`pipeline/run.py --compare`).

## 3. Antialiasing (SMAA 1x)

Escena de prueba con bordes duros (`upscale/aa_testscene.py`, 256x160):
rectángulos rotados y círculos. Referencia: la misma escena con
supersampling 8x8 (64 muestras por píxel).

| método | PSNR vs SSAA 8x8 |
|---|---|
| sin AA (1 muestra por píxel, como un juego sin AA) | 31.74 dB |
| RCAS (la nitidez de FSR1) | 30.47 dB |
| **SMAA 1x** | **38.37 dB** |

SMAA recupera la mayor parte de la diferencia con el supersampling; RCAS la
empeora porque realza el escalonado. En juegos que ya traen su propio AA
(FXAA/TAA) el beneficio será menor: se puede apagar por juego con `aa = 0`.

## 4. Verificación del camino de GPU

| prueba | resultado |
|---|---|
| runtime NumPy vs JAX (mismos pesos) | dif. máx. ~2e-6 |
| `host_test`: offsets de kernargs de los 19 kernels vs metadatos del compilador | todos coinciden |
| `host_test`: DCB 1080p / 900p→1080p / 720p→1080p | PM4 bien formado, 39-43 dispatches, 36-70 MiB |
| `gpu_emu` (DCB ejecutado con los kernels compilados para x86) vs runtime NumPy, 1280x720 | dif. máx. 1/255 (≈88 dB) |
| RCAS del camino GPU vs `upscale/fsr1.py` | ≈59 dB |
| SMAA del camino GPU (3 kernels) vs `upscale/smaa.py` | idéntico (0/255) |
| `host_test` con SMAA | 45-49 dispatches, 45-90 MiB de memoria de trabajo |
| plugin compilado con OpenOrbis + GoldHEN SDK | `vcfg_nx.prx`, exporta `plugin_load`/`plugin_unload` |
| PKG (`PkgTool.Core`) descifrado con `tools/pkg/pfs_extract.py` | eboot y prx idénticos a los compilados |
| edición de `plugins.ini` del instalador (test en PC) | respeta lo existente, idempotente |

Lo que **no** está verificado: ejecución en la GPU real de la PS4 (tiling de
los backbuffers, coherencia de caches, orden de los flips, rendimiento).

## 5. Ejemplo visual

Los tres recortes sintéticos donde NXFlow más mejora al block-matching
(+1,0 a +1,7 dB). Se ve que corrige los bordes en bloque y parte del
"fantasma" en oclusiones, pero los objetos muy rápidos siguen saliendo mal en
ambos métodos: ahí está el margen que queda por ganar con más entrenamiento.

![comparación](img/comparacion_sintetico.png)
