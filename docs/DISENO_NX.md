# VCFG-NX 2.2: diseño

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
        │     [SMAA 1x] [FSR1: EASU (si el juego renderiza < salida) + RCAS] en intermedio Y real
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
| SMAA 1x (real + generado) | 3 pases; la búsqueda solo en píxeles de borde | ~1-1,5 ms |
| **Total** | | **~4-6,5 ms** |

Son estimaciones a partir de 1,84 TFLOPS y del ancho de banda de la GPU de
PS4, no mediciones. La GPU de la consola ya está ocupada ~33 ms por el juego,
así que esto solo cabe en juegos que no usan el 100 % de la GPU; hay que
medirlo en consola (el plugin original mide sus pasadas con timestamps de GPU).

Memoria: 90 MiB de trabajo (1080p, FG + SMAA + RCAS) + 4 buffers de salida (33 MiB). Lo más pesado es la
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

## SMAA 1x

`shaders/smaa.cl` es una traducción de `SMAA.hlsl` a compute y a coordenadas
en píxeles (preset HIGH sin búsqueda diagonal, con detección de esquinas).
Usa las texturas precalculadas Area (160x560 RG8) y Search (64x16 R8)
originales, y guarda bordes y pesos en 8 bits como el SMAA original. Se
aplica sobre la imagen a resolución del juego, antes de EASU/RCAS, tanto al
fotograma generado como al real.


## Arquitectura del plugin (2.x)

```
main.c        hooks de GoldHEN -> llama al motor; implementa la "plataforma" con
              las funciones reales (memoria directa, VideoOut, Gnm, archivos)
nx_engine.c   toda la logica: estados, reservas, protecciones, menu, modos
nx_pipeline.c arma el DCB de un fotograma (FG, SMAA, EASU, NXSR/RCAS, OSD)
nx_osd.c      menu / contador / comparar con un contexto de pocos KiB
nx_dispatch.c despacho PM4 comun y la UNICA copia de la tabla de shaders
```

Como el motor no llama al sistema directamente, `plugin/test/console_sim.c`
lo ejecuta con una consola simulada (ver "Evaluación segura").

### Estados

| estado | qué hace |
|---|---|
| `WAIT_BUFFERS` | el juego todavía no registró sus framebuffers: todo pasa intacto |
| `WARMUP` | el juego está cargando: **no se reserva nada** durante `warmup_flips` (600 ≈ 20 s) |
| `RUN` | listo; los efectos solo si hubo memoria |
| `DEGRADED` | ni sin efectos da abasto la GPU (se reintenta a los ~3 min, 2 veces) o el juego cambió la geometría de sus buffers: solo pasa los fotogramas |
| `OFF` | formato no soportado (HDR), proceso que no es un juego: sin intervenir |

### Protecciones para que el juego no se cierre

1. **Arranque sin reservas.** Ni `plugin_load` ni el hook de `RegisterBuffers`
   piden memoria: se espera a que el juego termine de cargar.
2. **Reserva con margen.** Antes de pedir la memoria de los efectos (~125 MiB a
   1080p) se mide el bloque libre más grande y se exige dejar `reserve_mb`
   (192 MiB por defecto) para el juego. Si no alcanza, se reintenta 4 veces y
   luego los efectos quedan apagados con un aviso. El menú usa una reserva
   aparte de pocos KiB, así que sigue disponible.
3. **Modo seguro.** Antes de activar efectos se escribe
   `/data/vcfg_nx/guard-<TITLE>.txt` = `pending`; tras ~1 minuto estable, `ok`.
   Si el juego se cerró con `pending`, la sesión siguiente arranca sin
   efectos (modo seguro) y lo avisa; el usuario puede reactivarlos desde el
   menú. Con 3 cierres seguidos queda en modo seguro hasta reactivarlo.
4. **GPU acotada + calidad dinámica.** Se espera como mucho 4 ms a que termine
   el trabajo del fotograma anterior; si no terminó, ese fotograma va intacto.
   Cada atraso (o 15 fotogramas seguidos esperando más de 1,5 ms, que ya
   frenan al juego) baja **un escalón**, del más caro al más barato:

   | nivel | qué se apaga |
   |---|---|
   | -1 | NXSR Calidad → Rápido (si estaba en Calidad) |
   | -2 | SMAA |
   | -3 | NXSR / RCAS |
   | -4 | generación de fotogramas |

   Solo cuentan los escalones en uso. Entre dos bajadas hay 30 fotogramas de
   espera. Tras ~1 minuto sin atrasos prueba subir uno; si en los 20 s
   siguientes vuelve a atrasarse, la subida falló y la próxima espera el doble
   (hasta ~4 minutos). Cambiar algo en el menú vuelve a empezar desde arriba.
   Solo si con todo apagado sigue habiendo 30 atrasos seguidos (por ejemplo
   el contador de FPS con la GPU colgada) se pasa a `DEGRADED`.
5. **Re-registro de buffers.** Si el juego vuelve a registrar sus buffers con
   la misma geometría (pasa al volver del reposo o al cambiar opciones), se
   adoptan las nuevas direcciones y se sigue. Si todavía no había efectos
   activos, cualquier geometría vale: se empieza de nuevo. Con otra geometría
   y efectos ya activos (nuestros buffers de salida tienen el tamaño viejo) →
   `DEGRADED`.
6. **Datos raros → intacto.** Otro handle de VideoOut, índice fuera de rango,
   `count` 0 o enorme, HDR: el fotograma del juego se envía tal cual.
7. **Solo hooks, sin parches.** No se modifica memoria del juego; todo lo que
   se escribe en disco está en `/data/vcfg_nx` (y `plugins.ini` desde la app,
   con copia de seguridad y verificación de lo escrito).

## NXSR (reescalado / realce IA)

Ver `sr/net_sr.py`. Cabeza 3x3 (12→16) sobre el fotograma reordenado a media
resolución (space-to-depth 2), una capa 3x3 (16→16) con residual y cola 1x1
(16→12) que vuelve a resolución completa: ~1056 MAC por píxel de salida.

El **limitador** acota cada canal al mínimo/máximo de los 3x3 vecinos del
fotograma de entrada. Así el residuo puede empinar bordes y limpiar ruido,
pero no inventar valores nuevos: no hay halos ni grano por construcción. En
el barrido de arquitecturas, el limitador estricto rindió **mejor** que
relajarlo o quitarlo (funciona como un buen prior).

En la GPU (`shaders/nxsr.cl`) la red corre en un solo pase: cada grupo de 8x8
calcula la cabeza para su bloque + 1 de halo en memoria local (LDS, 6,4 KiB),
y el cuerpo, la cola y el limitador en registros. No hay buffers intermedios
a pantalla completa (GNMScale advierte que su versión no fusionada pedía
47 MiB). `tools/gcn_pack.py` escribe el tamaño de LDS en `COMPUTE_PGM_RSRC2`,
que en GNM lo programa el plugin y no el paquete de despacho.

Con FG activo y misma resolución de salida, NXSR (y SMAA) se aplican **una
vez** al fotograma real y la generación de fotogramas interpola entre reales
ya procesados: el intermedio hereda la calidad sin duplicar el coste. El
modelo de FG no cambió.

### Dos modelos: Rápido y Calidad

El mismo `shaders/nxsr.cl` se compila dos veces (`SR_C`, `SR_B`, `SR_NAME`):
`wg_nxsr` (Rápido) y `wg_nxsr_hq` (Calidad, dos capas de cuerpo: la primera
se calcula con halo en una segunda zona de LDS). `sr_model = 0` (Auto) usa
Calidad cuando el juego corre en modo PS4 Pro (`sceKernelIsNeoMode`), que
tiene 2,3 veces la GPU de una PS4; en una PS4 normal, Rápido. Si Calidad no
entra en el presupuesto, el primer escalón de la calidad dinámica vuelve a
Rápido.

Entrenamiento: cada conv lleva en paralelo una rama 1x1 (y la identidad en
las C→C) que se pliegan exactamente en el 3x3 al exportar
(`net_sr.fold`, re-parametrización estilo RepVGG/ECBSR): más capacidad al
entrenar, el mismo coste en la consola. Una pérdida de consistencia ante
desplazamientos de 1 píxel castiga que la salida cambie según la fase de la
celda 2x2 (lo que en movimiento se vería como parpadeo).

### Memoria

`nx_pipeline.c` reparte una sola arena. Lo que pasa de un fotograma al
siguiente (pirámide de luma y pooling del fotograma anterior, el intermedio
`tmp_img`) va aparte; los temporales del FG (flujo, entradas de nivel, capas
ocultas) y los del post-proceso (SMAA, intermedios de EASU/SMAA) se usan en
fases distintas del mismo trabajo, separadas por barreras, así que comparten
una región del tamaño del mayor de los dos grupos. El tamaño se calcula con el
mismo código que reparte (una pasada "en seco"), así no pueden desincronizarse.

| caso | antes (2.1) | ahora |
|---|---|---|
| 1080p FG + SMAA + NXSR | 90,2 MiB | 62,6 MiB |
| 900p → 1080p | 66,2 MiB | 45,5 MiB |
| 720p → 1080p | 45,3 MiB | 32,2 MiB |

Más los 4 buffers de salida (2 intermedios + 2 reales, 33 MiB a 1080p).

## App (instalador)

`installer/source/main.c` dibuja en la CPU sobre dos framebuffers de VideoOut
y solo redibuja cuando cambia algo (sin tocar nada, duerme mirando el mando).
Los íconos (`/user/appmeta/<ID>/icon0.png`, 512x512) se decodifican con
stb_image solo cuando se ven, se reducen a 176x176 por promedio de área y se
guardan los últimos 32. Los ajustes se leen y escriben con
`plugin/source/config.c`, el mismo código que usa el plugin.

## Evaluación segura (sin consola)

| prueba | qué valida |
|---|---|
| `plugin/test/console_sim` (ASan + UBSan) | 16 escenarios: arranque, poca memoria, GPU trabada (con y sin trabajo propio: pausa y reintento), cierre → modo seguro → reactivar, menú, HDR, juego a 60 fps, re-registro (otra geometría, misma geometría, antes de activar), ForWorkload + 2000 llamadas aleatorias, flips de CPU, GPU justa con modelo de tiempos (calidad dinámica baja y vuelve a subir), modelo Auto/Calidad en Pro, proceso que no es juego |
| `plugin/test/host_test` | offsets de argumentos de los 26 kernels vs compilador; DCB bien formado en 6 configuraciones |
| `test_gpu_path.py`, `test_smaa_path.py`, `test_sr_path.py` | el DCB ejecutado (kernels compilados para x86) coincide con las referencias: FG, SMAA y NXSR con LDS y barreras |
| `installer/test/test_core.c` (ASan) | edición de `plugins.ini` (conserva lo existente, idempotente, desinstalar), `param.sfo`, discos externos sin repetir, ajustes por juego ida y vuelta |
| `installer/test/test_app.c` (ASan + UBSan) | la app entera con un mando guionado sobre un sistema de archivos falso: activar, ajustes propios, volver a por defecto, cambiar los generales, guardar, salir sin guardar, desinstalar, consola sin juegos |
| regresión del emulador de GPU | con la memoria compartida, los 6 modos (FG, FG+RCAS, FG+SMAA, NXSR, FG+NXSR, NXSR Calidad) dan la misma salida byte a byte que antes |

## Qué falta validar en una consola

Marcado con `VALIDAR` en `plugin/source/main.c` y `gcn_desc.h`:

1. Índice de *tile mode* de los backbuffers (`NX_TILE_DISPLAY_2D_THIN`).
2. Que los dos flips por fotograma (`SubmitAndFlip` del intermedio + flip del
   real) se presenten en orden y en vblanks consecutivos en todos los juegos.
3. Bits exactos de `EVENT_WRITE_EOP` / `SURFACE_SYNC` para el flush de caches
   antes de presentar.
4. Juegos que presentan con `sceVideoOutSubmitFlip` desde la CPU: desde la 2.0
   se enganchan (nuestro DCB se envía y luego se llama al flip original), y el
   simulador lo cubre, pero el orden respecto del trabajo de GPU del juego hay
   que confirmarlo en consola. Colas de compute asíncronas: sin soporte.

## Próximos pasos (fase 2)

* **Escalado temporal tipo DLSS/FSR2** cuando haya vectores de movimiento del
  juego: acumular fotogramas anteriores reproyectados. Requiere encontrar el
  buffer de velocidad de cada juego (lo que intenta "GV" del original).
* Convoluciones con LDS también en NXFlow (NXSR ya las usa).
* Detección de cortes de escena: hoy en un corte se genera un intermedio
  "fantasma" (dura 16 ms). Se puede detectar con el coste del block-matching
  y repetir el fotograma.
