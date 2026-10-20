# VCFG-NX: diseño

Rediseño del frame generator VCFG para PS4 (GoldHEN). Objetivo: 30 → 60 fps
con menos artefactos que el block-matching puro, más un escalado/nitidez tipo
FSR, con un coste de GPU de pocos milisegundos por fotograma generado.

## Pipeline por fotograma del juego

```
 juego: SubmitAndFlip(buffer N)
        │
        ├─ 1. DCB del juego (sin flip)
        │
        ├─ 2. DCB de VCFG-NX  (39 dispatches a 1080p)
        │     luma + pirámide (3 niveles)          ← se reutiliza la del fotograma N-1
        │     RGB promedio 8x8 y 4x4               ← idem
        │     block-matching simétrico 3 niveles + mediana 3x3   → vectores 1/8
        │     NXFlow nivel A (1/8, interno 1/16): corrige flujo t→0, t→1 y predice máscara
        │     NXFlow nivel B (1/8, interno 1/16): residuo
        │     NXFlow nivel C (1/4, interno 1/8, 16 canales): residuo fino y máscara a 4 px
        │     warp_blend 1080p: backward-warp de N-1 y N con el sampler bilineal + mezcla
        │     [FSR1: EASU (si el juego renderiza < salida) + RCAS] en intermedio Y real
        │     → flip del intermedio (VSYNC)
        │
        └─ 3. flip del real (VSYNC, un vblank después)
```

### Por qué híbrido (block-matching + red)

Primero se entrenó una red pura (estilo RIFE reducido) que estimaba el flujo
sola. Con el presupuesto de entrenamiento disponible (CPU) no superó a la
mezcla 50/50. El híbrido arranca exactamente igual que el enfoque clásico
(con salida cero de la red el resultado es el block-matching de VCFG) y la red
solo aprende lo que al clásico le falta: corregir vectores, decidir de qué
fotograma tomar cada píxel en oclusiones (máscara) y mantener quieto el HUD.
Así nunca rinde peor que el original y el entrenamiento converge rápido.

### Coste estimado en PS4 (1080p)

| Etapa | Trabajo | Estimación |
|---|---|---|
| luma + pirámide + pool | 2 M píxeles, solo el fotograma nuevo | ~0,2 ms |
| block-matching | 3 niveles × 49 candidatos × SAD 8x8 | ~0,5 ms |
| NXFlow (17 conv 3x3: A y B 32 canales a 1/16, C 16 canales a 1/8) | ~1,25 GMAC | 1,5-3,5 ms (según eficiencia) |
| warp + mezcla 1080p | 2 muestras bilineales por píxel | ~0,3 ms |
| RCAS (real + generado) | 5 lecturas por píxel ×2 | ~0,3 ms |
| **Total** | | **~3-5 ms** |

Son estimaciones a partir de 1,84 TFLOPS y del ancho de banda de la GPU de
PS4, no mediciones. La GPU de la consola ya está ocupada ~33 ms por el juego,
así que esto solo cabe en juegos que no usan el 100 % de la GPU; hay que
medirlo en consola (el plugin original mide sus pasadas con timestamps de GPU).

Memoria: 70 MiB (1080p, FG + RCAS) + 4 buffers de salida. Lo más pesado es la
pirámide de luma en float32 de dos fotogramas (22 MiB); pasarla a 8/16 bits es
la optimización obvia.

## Del `.cl` a un shader de GNM

1. `shaders/*.cl` se compilan con clang al target `amdgcn-amd-amdhsa`,
   `-mcpu=gfx701` (Sea Islands, misma familia que la GPU "Liverpool").
2. No se usa ningún builtin que requiera las device-libs de ROCm: los ids
   salen de `__builtin_amdgcn_workgroup_id_*` / `workitem_id_*` con grupos fijos
   de 8x8, y las imágenes se leen con intrínsecos `llvm.amdgcn.image.*` sobre
   descriptores T#/S# crudos (`shaders/gcn_image.ll`), igual que un shader de
   GNM. Así se leen los backbuffers del juego con su tiling nativo.
3. `tools/gcn_pack.py` extrae de cada code object el código, los registros
   `COMPUTE_PGM_RSRC1/2`, el layout de user SGPRs y los offsets de cada
   argumento, y genera `plugin/include/nx_shaders.h`.
4. Todos los kernels usan el mismo ABI: 6 user SGPRs (`s[0:3]` buffer privado,
   sin scratch; `s[4:5]` puntero a kernargs). El plugin escribe
   `COMPUTE_USER_DATA_4/5` con la dirección del bloque de argumentos y despacha
   con `DISPATCH_DIRECT` (`plugin/source/pm4.c`).
5. `plugin/test/host_test.c` verifica en el PC que los structs de argumentos
   del plugin coinciden byte a byte con lo que espera cada kernel y que el DCB
   generado es PM4 bien formado.
6. `plugin/test/gpu_emu.c` **ejecuta el DCB**: interpreta los paquetes PM4,
   resuelve qué kernel apunta `COMPUTE_PGM_LO`, lee los kernargs y los T#/S#
   de la memoria y corre los mismos `.cl` compilados para x86.
   `test_gpu_path.py` compara el resultado con `model/reference_runtime.py`
   sobre fotogramas reales 1280x720: la diferencia máxima es 1/255 (redondeo
   a 8 bits) y el RCAS coincide con `upscale/fsr1.py`. Esto valida la lógica de
   los kernels, el orden de los pases, los offsets y los descriptores; lo que
   NO valida es el comportamiento del hardware real (tiling, caches, flips).

## Entrenamiento

* Datos (`model/data.py`): tripletes `(f0, f½, f1)` de video real (Big Buck
  Bunny, que es animación por computadora, y otros clips) más escenas
  sintéticas "tipo juego": fondo con paneo/zoom/giro de cámara, objetos que se
  tapan entre sí, HUD fijo semitransparente y fundidos. El sintético tiene
  ground truth exacto en t = ½.
* Pérdida: Charbonnier en píxeles + en gradientes (nitidez) + suavidad leve
  del flujo + **supervisión privilegiada de la máscara**: durante el
  entrenamiento se sabe cuál de los dos warps se parece más al fotograma real
  en cada píxel, y la máscara aprende a elegirlo (BCE). Un análisis oráculo
  mostró que una máscara perfecta valdría ~+2 dB sobre el block-matching, y
  sin esta señal la red aprendía muy poco de eso.
* Aumentos: espejo horizontal/vertical, inversión temporal y permutación de
  canales (el modelo no depende del orden RGB/BGR del backbuffer).
* `model/reference_runtime.py` reproduce la red en NumPy con la misma
  semántica que los shaders (padding SAME de XLA, resize half-pixel); coincide
  con JAX a ~1e-6.

Para mejorar el modelo, lo más rentable es entrenar con capturas reales del
juego objetivo: grabar gameplay a 60 fps (o 30 fps), pasarlo por
`model/data.py --videos ...` y reentrenar.

## Qué falta validar en una consola

Marcado con `VALIDAR` en `plugin/source/main.c` y `gcn_desc.h`:

1. Índice de *tile mode* de los backbuffers (`NX_TILE_DISPLAY_2D_THIN`).
2. Que los dos flips por fotograma (`SubmitAndFlip` del intermedio + flip del
   real) se presenten en orden y en vblanks consecutivos en todos los juegos.
3. Bits exactos de `EVENT_WRITE_EOP` / `SURFACE_SYNC` para el flush de caches
   antes de presentar.
4. Juegos que usan colas de compute asíncronas o `SubmitFlip` desde CPU: el
   original los detecta y se pone "inerte"; aquí quedan sin hook (nativo).

## Próximos pasos (fase 2)

* **Escalado temporal tipo DLSS/FSR2** cuando haya vectores de movimiento del
  juego: acumular fotogramas anteriores reproyectados. Requiere encontrar el
  buffer de velocidad de cada juego (lo que intenta "GV" del original).
* Convoluciones con LDS (tiles 16x16 + halo) para bajar el ancho de banda.
* Menú en pantalla (el original lo tiene; acá solo L3+R3 activar/desactivar y
  `/data/vcfg_nx/config.ini`).
* Detección de cortes de escena: hoy en un corte se genera un intermedio
  "fantasma" (dura 16 ms). Se puede detectar con el coste del block-matching
  y repetir el fotograma.
