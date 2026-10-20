# VCFG-NX: frame generation + FSR para PS4 (GoldHEN)

Rediseño del **VCFG v2.14**, el generador de fotogramas en beta que circula en
la comunidad de GoldHEN. Convierte juegos de 30 fps en 60 fps generando un
fotograma intermedio entre cada par, y agrega un escalado/nitidez tipo FSR.

| | VCFG v2.14 (original) | VCFG-NX (este repo) |
|---|---|---|
| Movimiento | block-matching + mediana | block-matching + **red NXFlow entrenada** que corrige vectores y predice oclusiones |
| Oclusiones | mezcla 50/50 | máscara aprendida (toma cada píxel del fotograma donde es visible) |
| HUD | rectángulo manual | aprendido (datos sintéticos con HUD fijo) |
| Escalado / nitidez | RCAS solo en el generado | **FSR1 completo** (EASU + RCAS) en real y generado |
| Shaders | arrays de ISA GCN a mano | OpenCL C → ISA GCN gfx701 con clang, empaquetado automático |

> **Estado:** el modelo está entrenado y verificado en PC; los shaders compilan
> a ISA de PS4 y el armado de command buffers se testea en PC. El plugin
> **todavía no se probó en una consola**: ver "Qué falta" en
> [`docs/DISENO_NX.md`](docs/DISENO_NX.md).

## Contenido

```
docs/ANALISIS_VCFG_v2.14.md   qué hay dentro del PKG original y cómo funciona
docs/DISENO_NX.md             arquitectura nueva, costes, ABI de los shaders, pendientes
docs/resultados.md            métricas: original vs nuevo
tools/pkg/                    extraer un fPKG: tabla de entradas, PFS, fSELF -> ELF
tools/gcn_pack.py             code object AMDGPU -> cabecera C para el plugin
model/                        datos, red (JAX), entrenamiento, evaluación, export
upscale/fsr1.py               port de referencia de AMD FSR1 (EASU + RCAS)
pipeline/run.py               demo en PC: video 30 fps -> 60 fps + FSR1
shaders/                      kernels de GPU (.cl) y script de compilación a GCN
plugin/                       plugin GoldHEN (C): hooks, PM4, descriptores, pipeline
plugin/test/                  test en PC del armado del DCB y del ABI de kernargs
weights/nxflow.pkl            pesos entrenados
```

## Uso rápido (PC)

```sh
pip install "jax[cpu]" optax pillow pyelftools pyyaml
# interpolar un video y ver original (izq.) vs nuevo (der.):
python pipeline/run.py entrada.mp4 salida_60fps.mp4 --compare --render-scale 0.75 --post easu
# evaluar:
python model/eval.py --weights weights/nxflow.pkl --sets val_video.npy val_synth.npy
```

Reentrenar con capturas propias (lo que más mejora la calidad en un juego concreto):

```sh
python model/data.py --videos gameplay1.mp4 gameplay2.mp4 --n-video 4000 --n-synth 2000 --crop 192 --out train.npy
python model/precompute_bm.py train.npy train_bm.npy        # idem para val.npy
python model/train.py --data train.npy --val val.npy --steps 15000 --out weights/nxflow.pkl
python model/export.py weights/nxflow.pkl plugin/include/nxflow_weights.h
```

## Compilar para PS4

```sh
shaders/build_gcn.sh                      # .cl -> ISA gfx701 -> plugin/include/nx_shaders.h
make -C plugin/test run                   # test en PC (ABI de kernargs + PM4)
# plugin: copiar plugin/ a GoldHEN_Plugins_Repository/plugin_src/vcfg_nx y
make -C GoldHEN_Plugins_Repository/plugin_src/vcfg_nx   # requiere OpenOrbis + GoldHEN SDK
```

Instalar `vcfg_nx.prx` en `/data/GoldHEN/plugins/` y agregarlo a `plugins.ini`
bajo el Title ID del juego. Configuración en `/data/vcfg_nx/config.ini`
(o `/data/vcfg_nx/<TITLE_ID>.ini`):

```ini
framegen = 1        ; 30 -> 60
post = 1            ; 0 nada, 1 RCAS, 2 FSR1 (EASU+RCAS) si el juego renderiza < salida
sharpness = 0.25    ; 0 = máximo, 2 = suave
max_game_fps = 48   ; por encima, FG se apaga solo
toggle_hold_ms = 1500 ; L3+R3 mantenido -> on/off
```

## Créditos y licencias

* VCFG original: autor/es de la comunidad GoldHEN. Este repo no incluye sus
  binarios; solo el análisis.
* FSR1: AMD FidelityFX Super Resolution, licencia MIT (port en `upscale/` y `shaders/fsr1.cl`).
* Formato PKG/PFS: documentado por LibOrbisPkg (maxton).
* `plugin/Makefile` deriva de GoldHEN_Plugins_Repository (GPL-3.0).
* Datos de entrenamiento: Big Buck Bunny (© Blender Foundation, CC-BY 3.0),
  clips de muestra de scikit-video y OpenCV, y escenas sintéticas propias.
