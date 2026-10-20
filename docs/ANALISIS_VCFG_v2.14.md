# Análisis de VCFG v2.14 (PKG de la comunidad GoldHEN)

Archivo analizado: `VCFG_v2.14.pkg` (6,3 MB). Todo lo de abajo sale de abrir el
PKG con las herramientas de `tools/pkg/` y de leer los binarios (strings,
desensamblado). No se ejecutó en una consola.

## 1. El paquete

| Campo | Valor |
|---|---|
| Content ID | `IV0000-VCFG00001_00-VCFGINSTALLER000` |
| Title / Title ID | `VCFG` / `VCFG00001` |
| Versión | 2.14 (`c_date=20261019`) |
| Tipo | fPKG (PKG falso) con *passcode* por defecto (32 ceros) |
| PFS | cifrado XTS-AES-128 con clave derivada del passcode, *seed* en cero; imagen interna comprimida (PFSC) |

Cómo se extrajo (reproducible):

```sh
python3 tools/pkg/pkginfo.py VCFG_v2.14.pkg        # tabla de entradas del PKG
python3 tools/pkg/pfs_extract.py VCFG_v2.14.pkg out/  # descifra PFS y extrae los archivos
python3 tools/pkg/unself.py out/eboot.bin eboot.elf   # fSELF -> ELF
```

Contenido:

| Archivo | Tamaño | Qué es |
|---|---|---|
| `eboot.bin` | 221 KB | Lanzador/instalador (app con UI propia) |
| `plugins/vcfg10.prx` | 495 KB | Plugin principal ("REF", modo 8.6) |
| `plugins/vcfg_classic18.prx` | 409 KB | Variante "classic" del plugin |
| `assets/fonts/*` | ~1,1 MB | Inter y JetBrains Mono (OFL) |
| `sce_module/libc.prx`, `libSceFios2.prx` | | módulos estándar de homebrew |

Los tres ejecutables son *fake SELF* sin cifrar (`4F 15 3D 1D`), se convierten
a ELF directamente. Las rutas de compilación que quedaron en el binario
(`C:\Users\user\Documents\Codex\2026-10-18\new-chat\work\vcfg\...`) indican que
el código se generó con un asistente de IA en un par de días.

## 2. El lanzador (`eboot.bin`)

* Abre VideoOut directamente y dibuja su propia UI (stb_image + stb_truetype).
* Lista `/user/app`, cruza con una lista interna de juegos soportados
  (`CUSA07439`, `CUSA08519`, `CUSA00411`, `CUSA00207`) y muestra los instalados.
  El prefijo de todos los logs, `bb_probe`, sugiere que el objetivo principal
  es Bloodborne.
* Copia los PRX a `/data/GoldHEN/plugins/` y reescribe `plugins.ini`
  (con copia de seguridad `plugins.ini.vcfg_backup_<timestamp>`).
* Guarda ajustes en `/data/bb_probe/vcfg_launcher.ini` y genera un informe en
  `/data/vcfg_report.txt` con el log del plugin de las últimas partidas.

## 3. El plugin (`vcfg10.prx`)

### Hooks (funciones del sistema interceptadas)

`sceGnmSubmitAndFlipCommandBuffers(+ForWorkload)`, `sceGnmSubmitCommandBuffers(+ForWorkload)`,
`sceVideoOutSubmitFlip`, `sceVideoOutRegisterBuffers`, `sceVideoOutSetFlipRate`,
`sceVideoOutAddFlipEvent`, `sceVideoOutGetFlipStatus`, `sceKernelWaitEqueue`,
`sceKernelGetEventData`, `sceKernelAllocateDirectMemory`, `sceKernelMapDirectMemory`,
`sceKernelMapNamedDirectMemory` y las lecturas de mando `scePadRead*`.

### Cómo genera los fotogramas (según los strings de telemetría)

```
PASSES ms: pyr8 | me3 | me2 | gm8+fgs8 | me1 | mf8 | fence | phase8 | warp8+rcas+osd
```

1. **pyr8**: pirámide de luminancia.
2. **me3 → me2 → me1**: estimación de movimiento por bloques, de grueso a fino.
3. **gm8 / fgs8**: movimiento global de cámara y separación de primer plano.
4. **mf8**: filtro de mediana de los vectores.
5. **warp8**: deformación y mezcla de los dos fotogramas; luego **RCAS** (el
   sharpening de AMD FSR) y el **OSD**.

Modo de presentación "2x": por cada fotograma del juego hace dos flips (el
intermedio y el real) forzando la salida a 60 Hz con `SetFlipRate(0)`.

Además intenta usar los **vectores de movimiento del propio juego** ("GV"):
recorre los command buffers del juego buscando render targets de 2 canales
que parezcan un buffer de velocidad, calibra su codificación (coeficientes
`a`, `b`, R²) contra el flujo óptico y guarda un perfil por juego en
`/data/bb_probe/profiles/`. Si no concuerda, cae a "solo flujo óptico".

Los shaders son arrays de ISA GCN armados a mano ("A13_exact_shader_arrays");
se identifican unos 30 bloques de código.

### Otras funciones

* Menú con L3+R3 (3 s): frame generation, preset, nitidez, "enhance" (AA
  direccional), overlay de FPS, opacidad, `hud_protect` (rectángulo manual).
* *Crash guard*: si la partida anterior no terminó bien arranca en modo pasivo.
* *Memory valve*: se pausa si el juego necesita la memoria directa.
* Se desactiva solo si el juego ya corre a ≥ 48 fps.

## 4. Problemas encontrados

| Problema | Consecuencia |
|---|---|
| Movimiento solo por bloques + mediana, mezcla 50/50 | bordes en bloque, "halos" y fantasmas en oclusiones (lo que se tapa/destapa) |
| HUD protegido con un rectángulo manual | menús, subtítulos y marcadores tiemblan o se deforman |
| Detección de vectores del juego por heurística | frágil y específica de cada juego; mucho código y tiempo de CPU en el hilo de render |
| Telemetría de depuración en la versión final | decenas de archivos y CSV escritos desde el proceso del juego (`/data/bb_probe/*`) |
| Shaders como arrays de ISA a mano | imposible de mantener o auditar |
| RCAS solo en el fotograma generado | la nitidez alterna entre fotogramas (parpadeo de detalle) |

## 5. Medición del enfoque original

`model/baseline_bm.py` reimplementa la parte central (pirámide + búsqueda por
bloques + mediana + warp simétrico). En los datos de validación mejora entre
1,3 y 2,7 dB de PSNR sobre la mezcla 50/50: la base técnica del original es
sólida. El modelo nuevo apenas lo supera por ahora (ver `docs/resultados.md`);
las mejoras más claras del rediseño están en el resto del sistema (FSR1 en
todos los fotogramas, shaders generados desde código fuente y verificados,
plugin sin telemetría de depuración).
