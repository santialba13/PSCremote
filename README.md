# VCFG-NX: frame generation + SMAA + FSR1 para PS4 (GoldHEN)

Rediseño del **VCFG v2.14**, el generador de fotogramas en beta que circula en
la comunidad de GoldHEN. Convierte juegos de 30 fps en 60 fps generando un
fotograma intermedio entre cada par, suaviza bordes con **SMAA 1x** y agrega
nitidez/escalado tipo **FSR1**.

**Descarga:** [`dist/VCFG-NX_v1.00.pkg`](dist/VCFG-NX_v1.00.pkg) (fPKG, instalador) ·
[`dist/vcfg_nx.prx`](dist/vcfg_nx.prx) (plugin suelto) · [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt)

| | VCFG v2.14 (original) | VCFG-NX (este repo) |
|---|---|---|
| Movimiento | block-matching + mediana | block-matching + **red NXFlow entrenada** que corrige vectores y predice oclusiones |
| Oclusiones | mezcla 50/50 | máscara aprendida (toma cada píxel del fotograma donde es visible) |
| HUD | rectángulo manual | aprendido (datos sintéticos con HUD fijo) |
| Antialiasing | "edge AA / directional AA" | **SMAA 1x** (preset HIGH, esquinas) en real y generado |
| Escalado / nitidez | RCAS solo en el generado | **FSR1** (EASU + RCAS) en real y generado |
| Shaders | arrays de ISA GCN a mano | OpenCL C → ISA GCN gfx701 con clang, empaquetado automático |

> **Estado — leer antes de instalar.** Todo compila con el toolchain real
> (OpenOrbis + GoldHEN SDK) y el PKG se genera y verifica en PC, pero **nada
> se ejecutó todavía en una PS4**. Es una beta de prueba: puede no funcionar o
> colgar el juego. Lo verificado y lo pendiente está en
> [`docs/resultados.md`](docs/resultados.md) y [`docs/DISENO_NX.md`](docs/DISENO_NX.md).
>
> El modelo de frame generation, entrenado solo en CPU, mejora muy poco al
> block-matching del original (+0,16 dB en escenas tipo juego, empate en
> video). SMAA sí mejora claramente los bordes (+6,6 dB contra SSAA 8x8,
> donde FSR1/RCAS empeora −1,3 dB).

## ¿Y DLSS (HelixSR)?

Se evaluó [HelixSR](https://github.com/lonewolf0622/HelixSR) (la red de DLSS
dentro de una DLL compatible con FSR 3.1). **No se puede usar en PS4**:

* es una DLL de **Windows/Direct3D 12** que se inyecta en juegos de PC; la PS4
  usa GNM y un sistema operativo propio;
* necesita **vectores de movimiento y profundidad** que el juego le entrega por
  la API de FSR/DLSS; los juegos de PS4 no exponen eso a un plugin;
* apunta a GPUs RDNA 1/2 (y la propia descripción duda de GCN4); la PS4 es
  GCN 1.1, varias generaciones más vieja;
* su licencia **prohíbe modificarla o descompilarla**, y los pesos de la red son
  de NVIDIA y **no se pueden redistribuir** (HelixSR los genera en la PC de cada
  usuario a partir de la DLL de DLSS). Meterlos en un PKG no es legal.

Por eso, como pediste para el caso de incompatibilidad, se agregó **SMAA 1x**,
un antialiasing morfológico mucho más fuerte que FSR1 (FSR1 escala y da
nitidez, pero no suaviza bordes) y viable en la GPU de la PS4.

## Instalar en la PS4

1. Requiere **GoldHEN** con soporte de plugins.
2. Instalar `VCFG-NX_v1.00.pkg` (Package Installer de GoldHEN) y abrir
   **VCFG-NX Installer**. Copia el plugin a `/data/GoldHEN/plugins/`, crea
   `/data/vcfg_nx/config.ini` y `/data/vcfg_nx/titles.txt`, y agrega el plugin
   a `/data/GoldHEN/plugins.ini` para los juegos de `titles.txt` que estén
   instalados (hace copia de seguridad del `plugins.ini`).
3. Para otros juegos: agregar su Title ID a `/data/vcfg_nx/titles.txt` y volver
   a abrir el instalador.
4. En el juego: mantener **L3+R3** 1,5 s para activar/desactivar.

Configuración (`/data/vcfg_nx/config.ini`, o `/data/vcfg_nx/<TITLE_ID>.ini` por juego):

```ini
framegen = 1        ; 30 -> 60
aa = 1              ; SMAA 1x en reales y generados
post = 1            ; 0 nada, 1 RCAS, 2 FSR1 (EASU+RCAS) si el juego renderiza < salida
sharpness = 0.25    ; 0 = máximo, 2 = suave
max_game_fps = 48   ; por encima, FG se apaga solo
toggle_hold_ms = 1500 ; L3+R3 mantenido -> on/off
```

## Contenido

```
dist/                         PKG instalador, plugin .prx y sumas SHA256
docs/ANALISIS_VCFG_v2.14.md   qué hay dentro del PKG original y cómo funciona
docs/DISENO_NX.md             arquitectura, costes, ABI de los shaders, pendientes
docs/resultados.md            métricas: FG, SMAA, FSR1 y verificación del camino GPU
tools/pkg/                    extraer un fPKG: tabla de entradas, PFS, fSELF -> ELF
tools/gcn_pack.py             code object AMDGPU -> cabecera C para el plugin
model/                        datos, red (JAX), entrenamiento, evaluación, export
upscale/                      referencias: FSR1 (fsr1.py), SMAA (smaa.py), escena de prueba de AA
pipeline/run.py               demo en PC: video 30 fps -> 60 fps + FSR1
shaders/                      kernels de GPU (.cl) y script de compilación a GCN
plugin/                       plugin GoldHEN (C): hooks, PM4, descriptores, pipeline
plugin/test/                  tests en PC: ABI de kernargs, PM4 y emulador que ejecuta el DCB
installer/                    app instaladora (eboot) y armado del PKG
third_party/smaa/             texturas precalculadas de SMAA (MIT)
weights/nxflow.pkl            pesos entrenados
```

## Compilar

```sh
pip install "jax[cpu]" optax pillow pyelftools pyyaml
shaders/build_gcn.sh                                   # .cl -> ISA gfx701 -> plugin/include/nx_shaders.h
make -C plugin/test all && plugin/test/host_test        # ABI de kernargs + PM4
python3 plugin/test/test_gpu_path.py video.mp4 weights/nxflow.pkl   # FG: DCB emulado vs modelo
python3 plugin/test/test_smaa_path.py                  # SMAA: DCB emulado vs referencia
# PRX y PKG (OpenOrbis v0.5.4 "toolchain-llvm-18" + GoldHEN_Plugins_SDK compilado):
make -C installer OO_PS4_TOOLCHAIN=... GOLDHEN_SDK=... [SSL11_DIR=carpeta con libssl.so.1.1]
```

`PkgTool.Core` (del toolchain) es .NET Core 3 y necesita **libssl 1.1**; en
distribuciones nuevas se puede extraer del `.deb` de Ubuntu 20.04 y pasar su
carpeta en `SSL11_DIR`.

Reentrenar con capturas propias (lo que más mejora la calidad en un juego concreto):

```sh
python model/data.py --videos gameplay1.mp4 gameplay2.mp4 --n-video 4000 --n-synth 2000 --crop 192 --out train.npy
python model/precompute_bm.py train.npy train_bm.npy        # idem para val.npy
python model/train.py --data train.npy --val val.npy --steps 15000 --out weights/nxflow.pkl
python model/export.py weights/nxflow.pkl plugin/include/nxflow_weights.h
```

## Créditos y licencias

* VCFG original: autor/es de la comunidad GoldHEN. Este repo no incluye sus
  binarios; solo el análisis.
* SMAA: Jimenez, Echevarria, Masia, Navarro, Gutierrez — licencia MIT
  (`third_party/smaa/LICENSE.txt`; port en `upscale/smaa.py` y `shaders/smaa.cl`).
* FSR1: AMD FidelityFX Super Resolution, licencia MIT (port en `upscale/fsr1.py` y `shaders/fsr1.cl`).
* Formato PKG/PFS: documentado por LibOrbisPkg (maxton).
* Toolchain: OpenOrbis PS4 Toolchain y GoldHEN Plugins SDK. El PKG incluye
  `libc.prx`, `libSceFios2.prx` y `right.sprx` tomados de los ejemplos de OpenOrbis.
* Datos de entrenamiento: Big Buck Bunny (© Blender Foundation, CC-BY 3.0),
  clips de muestra de scikit-video y OpenCV, y escenas sintéticas propias.
