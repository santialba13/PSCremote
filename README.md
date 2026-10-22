# VCFG-NX 2.76: frame generation + realismo IA (NXRE) + reescalado IA (NXSR) para PS4 (GoldHEN)

Rediseño del **VCFG v2.14** (generador de fotogramas en beta de la comunidad
de GoldHEN) que además incorpora un **modelo de reescalado/realce propio,
NXSR**, como alternativa a GNMScale pero sin el granulado que este agrega.
Todo con la interfaz de VCFG: app instaladora con lista de juegos y menú en
pantalla con **L3+R3**.

**Descarga:** [`dist/VCFG-NX_v2.76.pkg`](dist/VCFG-NX_v2.76.pkg) (fPKG, app) ·
[`dist/vcfg_nx.prx`](dist/vcfg_nx.prx) (plugin suelto) · [`dist/SHA256SUMS.txt`](dist/SHA256SUMS.txt)

| servicio | qué hace |
|---|---|
| **Generación de fotogramas** | 30 → 60 fps. **VCFG** (el motor clásico de VCFG, MIT) o **IA** (NXFG-2, propia: movimiento de 4 niveles, red chica, elección de vector por píxel y protecciones anti-parpadeo) |
| **Realismo IA (NXRE)** | red propia que cambia luz y color por zona y por nivel de brillo para que la imagen se vea como una foto real (aprendida de GTA V convertido a fotorrealista); ~1-2 ms a 1080p, sin parpadeos |
| **Reescalado IA (NXSR)** | red de 16 canales que reconstruye detalle (TAA borroso, render a menos resolución) y **nunca agrega grano ni halos**: su salida queda acotada al rango de los píxeles vecinos |
| SMAA 1x | antialiasing de bordes |
| FSR1 / **FSR+** | EASU si el juego renderiza por debajo de la salida; RCAS + realce de detalle sin halos como nitidez (también junto con NXSR y con la generación) |

> **Estado — leer antes de instalar.** Compila con el toolchain real
> (OpenOrbis + GoldHEN SDK), el PKG se genera y se verifica en PC, y toda la
> lógica del plugin se probó en un **simulador de consola** con detectores de
> errores de memoria. Pero **nada se ejecutó todavía en una PS4**: es una beta.
> Las protecciones de abajo existen para que, si algo falla, el juego siga
> como si el plugin no estuviera. Detalles en [`docs/resultados.md`](docs/resultados.md)
> y [`docs/DISENO_NX.md`](docs/DISENO_NX.md).

## NXSR frente a GNMScale

GNMScale ([análisis](docs/ANALISIS_GNMSCALE.md)) trae tres redes de 4 canales
que son filtros de nitidez aprendidos sin límite. Evaluadas sobre el mismo
conjunto (Urban100, BSD100, Set14 y Big Buck Bunny con degradaciones de
juego):

| método | PSNR | SSIM | PSNR en zonas planas (grano) | píxeles con halo |
|---|---|---|---|---|
| imagen sin procesar | 27.28 | 0.886 | 42.57 | 1.9 % |
| FSR1 RCAS | 26.79 | 0.868 | 35.59 | 18.2 % |
| GNMScale fidelity | 25.16 | 0.828 | 35.56 | 25.6 % |
| GNMScale sharp-fast | 24.92 | 0.829 | 35.71 | 26.2 % |
| GNMScale sharp | 25.11 | 0.825 | 35.52 | 26.5 % |
| **NXSR Rápido (VCFG-NX)** | **27.74** | **0.896** | **42.71** | **2.2 %** |
| **NXSR Calidad (VCFG-NX, PS4 Pro)** | **27.90** | **0.900** | **44.25** | **1.5 %** |

* GNMScale empeora la imagen: amplifica ruido en zonas planas (el granulado)
  y deja halos en uno de cada cuatro píxeles.
* NXSR mejora la imagen y mantiene zonas planas y bordes limpios. La mejora en
  PSNR es moderada: con ~1000 operaciones por píxel (lo que la GPU de la PS4
  puede pagar a 1080p junto con la generación de fotogramas) no se puede
  "inventar" mucho detalle; lo que gana es nitidez real sin artefactos.

## Menú dentro del juego (L3+R3)

![menú L3+R3](docs/img/menu_juego.jpg)

Mantener **L3+R3** 1,5 s abre el panel sobre el juego (el juego no recibe los
botones mientras está abierto):

* **Todos los efectos**: interruptor general; apaga o prende todo de una vez
  sin perder lo que elegiste en cada fila.
* **Modelo IA**: izquierda/derecha cambia entre **Ligero** (NXSR-L), **Clásico**
  (el NXSR Rápido de antes, caro) y **Calidad** (NXSR-L Calidad); con modo ahorro,
  "Ligero (ahorro)".
* **Generación**: **VCFG** (motor clásico de VCFG) o **IA NXFG-2** (la propia).
* **Realismo IA (NXRE)**: OFF / Suave / Normal / Fuerte (Normal por defecto).
* Generación de fotogramas, NXSR (Suave / Normal / Fuerte), modo ahorro, SMAA,
  nitidez FSR+, contador de FPS (con temperatura) y comparar (con línea divisoria).
* Arriba se ven los FPS, la temperatura y el estado (activo, calidad dinámica,
  calor). **Guardar** deja todo para ese juego; **O** cierra.

## Novedades de la 2.76

Pediste algo tipo **DLSS 5**: que la IA mejore cómo se ve el juego (realismo), no
solo la nitidez, que se note mucho y que entre todo junto sin "carga alta".

**Qué es DLSS 5 y qué se puede en una PS4.** DLSS 5 es un modelo generativo
(transformer de difusión en espacio de píxeles) que recibe el fotograma más datos
3D del motor (normales, profundidad, vectores de movimiento) y rehace la luz y los
materiales. Necesita una RTX de las grandes; la PS4 tiene ~1,8 TFLOPS y el juego usa
casi todos. Lo que sí entra es la idea de los trabajos de investigación sobre
fotorrealismo en GTA V (EPE de Intel, 2021, y REGEN, 2025): una red que convierte el
fotograma del juego al aspecto de fotos reales de calles (Cityscapes). REGEN publicó
un video con GTA V a la izquierda y su resultado fotorrealista a la derecha.

**NXRE ("realismo"), modelo propio** (`sr/net_re.py`, `shaders/nxre.cl`):

- Una red chica mira el fotograma entero a ~128 x 72 (4 capas de paso 2 + contexto
  global de la escena) y arma una **malla de transformaciones de color**: por zona de
  la imagen y por nivel de brillo (8 niveles), una matriz afín 3x4. Cada píxel toma
  la suya (bilineal en el espacio, lineal entre niveles de brillo) y la aplica. Es la
  idea de "Deep Bilateral Learning" (HDRNet, Google/MIT 2017), la forma más barata
  conocida de hacer una transformación aprendida de imagen completa.
- **Entrenada por destilación**: de ese video armé 3399 pares alineados (la mitad
  derecha está ampliada x1,236 y va 2 fotogramas atrasada: alineación automática con
  ECC sobre los bordes, `sr/make_re_pairs.py`). La red aprende a llevar GTA V al
  aspecto fotorrealista: luz más natural, cielos y sombras de cámara real, colores
  menos "de videojuego". No se usa ningún modelo de ellos: solo sus imágenes como
  ejemplo a imitar.
- Validación (fotogramas que no vio): pasa de **14,4 dB a ~23,9 dB** de parecido con
  el resultado fotorrealista. Cambio medio por píxel: ~27-35/255 (se nota mucho, a
  simple vista; NXSR-L cambiaba ~1/255).
- **No parpadea ni inventa texturas**: es una transformación de color suave por zonas
  (no puede generar artefactos de detalle), la malla además se promedia con la del
  fotograma anterior, y el intermedio de la generación se arma sobre el real ya
  procesado.
- Costo medido con el código máquina: **~14 M instrucciones a 1080p (~1-2 ms)**, la
  red en sí es casi nada (~1 M); el resto es aplicar la malla por píxel.
- Menú: **Realismo IA (NXRE): OFF / Suave / Normal / Fuerte** (intensidad 0,5 / 0,8
  / 1). Normal por defecto. Va primero en la cadena: después SMAA, NXSR-L y FSR+
  trabajan sobre la imagen final. También en la app (ajustes por juego).

Honestidad: aprendió de escenas de día manejando por Los Santos; en interiores,
noche o menús aplica la misma idea (lo probé con escenas oscurecidas y con otro tipo
de imagen: no rompe nada), pero donde más se luce es en exteriores. El estilo es el
de las fotos de Cityscapes: más sobrio y "de cámara" que el GTA original.

**Todo a la vez sin "carga alta".** Hasta la 2.75 la calidad dinámica bajaba efectos
si nuestro trabajo pasaba 8 ms, aunque el juego siguiera a 30 fps. Ahora solo baja si
el **juego de verdad pierde fotogramas** (fps del juego < 28,5 o el período de la GPU
entre fotogramas > 35 ms) o si lo nuestro pasa un límite duro de 14 ms. En el
simulador, con un juego de 20 ms de GPU y todo prendido (IA + NXRE + NXSR-L + SMAA +
FSR+, ~11,5 ms propios): nivel 0, sin aviso, 1000/1000 intermedios. Con un juego de
26 ms baja dos escalones y la generación sigue.

**Pruebas.** NXRE en el emulador de PC contra JAX (máx. 1/255) y en el emulador del
código máquina GCN (idéntico, también el modo "todo a la vez"); simulador de consola
con todos los escenarios (nuevo: 33, todo a la vez). El emulador GCN ahora soporta
`v_writelane/v_readlane` (la capa de salida de NXRE derrama SGPRs). **Sigue sin
probarse en una PS4.**

## Novedades de la 2.75

Con la 2.74 en GTA V: NXSR-L corría bien pero "no cambiaba nada", el modelo
Calidad tampoco, la generación **IA** se apagaba apenas se elegía ("carga alta") y
el antialiasing no se veía. Las cuatro cosas tenían la **misma causa**:

- **La carga de GPU se medía mal con la generación.** En la presentación de VCFG
  nuestro trabajo espera dentro de la GPU a que la pantalla muestre el fotograma
  intermedio (WaitFlipDone, ~8-16 ms). Esa espera se contaba como trabajo nuestro,
  así que cualquier efecto parecía "carga alta" y la calidad dinámica sacaba, en
  orden, SMAA → reescalado → nitidez → generación IA. Por eso nada se notaba.
  Ahora se marcan tiempos antes y después de la espera y se descuenta: se mide solo
  lo que hacemos (en el simulador: 2,8 ms propios con 9 ms de espera → nivel 0, todo
  activo). El registro muestra los dos (`espera=… ms`).
- **La calidad dinámica ya no apaga la generación** en ese camino: lo máximo que
  hace es pasar de IA a la de VCFG (avisa una vez en pantalla). Presupuesto propio:
  8 ms por fotograma del juego.

**NXSR-L más visible.** Suave / Normal / Fuerte = **2,0 / 3,0 / 4,5** (antes 1,2 /
2,0 / 3,0). El núcleo deja afuera el grano y el límite al rango 3x3 de los vecinos
sigue acotando los halos aun a 4,5.

**NXSR-L Calidad (modelo nuevo, entrenado para esta versión).** Misma estructura
que Ligero pero con 16 canales y una capa más en la red guía, entrenado dando el
doble de peso a los bordes y con más margen de realce (35 %). Validación: 27,95 dB
(Ligero 27,94), ~31 M instrucciones a 1080p contra ~307 M del Calidad viejo: entra
junto con la generación. El modelo "Calidad" viejo ya no se usa; el "Rápido" viejo
queda como **Clásico**.

**Generación IA NXFG-2** en el menú con ese nombre ("IA NXFG-2"). Es el modelo
propio (movimiento de 4 niveles, red chica, elección de vector por píxel, máscara
anti-fantasma del HUD, guarda de zonas quietas, corte de escena). Costo medido con
el código máquina: ~40 M instrucciones a 1080p (~3-6 ms), parecido al motor de VCFG.

**SMAA 2,5 veces más rápido y siempre activo si lo elegís.**

- El pase caro (búsqueda de líneas) leía los bordes de un buffer de bytes con el
  filtro bilineal hecho a mano (8 lecturas por muestra). Ahora los bordes son una
  imagen RGBA8 y se leen con el **filtro bilineal del hardware**, como el SMAA
  original (lo mismo las texturas de áreas y de búsqueda). Resultado idéntico a la
  referencia (0/255 de diferencia), ~32 M instrucciones a 1080p en vez de ~57 M.
- 8 pasos de búsqueda (preset Medium) y el modo ahorro ya no lo apaga.

**Nitidez FSR+ activada por defecto.**

**Pruebas.** Emulador de PC y emulador de instrucciones GCN: todos los modos
idénticos (IA, VCFG, SMAA, NXSR-L Ligero y Calidad, combinados). Simulador de
consola: todos los escenarios, incluidos los nuevos (espera de pantalla que no
cuenta como carga, IA + todo con la GPU justa sin que se apague la generación). Los
imports del plugin son los mismos que en la 2.74. **Sigue sin probarse en una PS4**:
los tiempos son estimados del código máquina, no medidos en la consola.

## Novedades de la 2.74

Con la 2.73 en GTA V: la generación de VCFG perfecta; el reescalado no se notaba
(salvo el modelo Rápido), con el modelo Calidad la carga obligaba a bajar de nivel,
el SMAA no se veía y la nitidez FSR se notaba poco.

**Modelo nuevo de reescalado: NXSR-L ("Ligero")**, pensado para ir junto con la
generación (`sr/net_srl.py`, `shaders/nxsrl.cl`):

- La imagen se separa en dos bandas de detalle de la luma (radios 1 y 2) con
  filtros fijos baratos, y una **red chica a 1/8 de resolución** (celdas de 8x8:
  color medio y energía de cada banda → 3 capas 3x3 de 12 canales) decide por zona
  cuánto realzar cada banda y desde qué umbral ignorar el grano. Un límite al
  rango 3x3 local (más un 25 %) evita los halos; solo toca la luma (sin bordes de
  color).
- Medido en el mismo conjunto de validación que los otros modelos:

  | modelo | PSNR | SSIM | zonas planas (grano) | instrucciones de GPU por fotograma 1080p |
  |---|---|---|---|---|
  | sin procesar | 27,37 | 0,888 | 41,98 | — |
  | NXSR Rápido | 27,84 | 0,898 | 42,19 | ~79 M |
  | NXSR Calidad | 27,90 | 0,900 | — | ~307 M |
  | **NXSR-L** | **27,92** | **0,904** | **42,03** | **~26 M** |

  Mejor que los dos anteriores y 3 veces más barato que Rápido (12 veces que
  Calidad). Las instrucciones son las del código máquina real, contadas con el
  emulador de la GPU.
- Es el modelo por defecto y el del modo ahorro. Suave / Normal / Fuerte = 1,2 /
  2,0 / 3,0: hasta 3 sigue mejorando la imagen (PSNR por encima del original) sin
  agregar grano, así que se usa fuerte para que se note a 1080p nativos.
- Menú: **Modelo IA: Ligero / Rápido / Calidad**. La calidad dinámica, si falta
  GPU, primero cambia Rápido o Calidad por Ligero; recién después saca SMAA, el
  reescalado y la nitidez. La generación queda siempre.

**SMAA y FSR+ más visibles.**

- El modo ahorro ya no apaga el SMAA (era la razón más probable de que no se
  viera); en ahorro solo se fuerza NXSR-L.
- SMAA con umbral de borde 0,05 (preset "Ultra"): los juegos ya traen FXAA/TAA y
  sus bordes quedan con poco contraste; con 0,1 casi no los encontraba.
- FSR+ más fuerte: Baja / Media / Alta = detalle 0,5 / 1,0 / 1,6, con un
  sobreimpulso permitido que crece con la intensidad.

**Lo que se tomó de mirar GNMScale** (la versión nueva que pasaste; no se usa
ninguno de sus modelos, ver `docs/ANALISIS_GNMSCALE.md`): su integración es la
misma que la nuestra (trabajo al final de los comandos del juego, resultado
copiado al buffer del juego, solo SDR de 8 bits). Lo que hace que su efecto se
note es la intensidad (corre la red a 1/N de la resolución y reconstruye, con
"textura generada" e intensidad por encima de 1, que ellos mismos avisan que hace
halos y tiembla). Su FBSR usa la misma idea que NXSR-L: celdas de 8x8 con
estadísticas y una red de contexto que modula un filtro por píxel.

**Pruebas.** NXSR-L en el emulador de PC (contra JAX: 1/255) y en el emulador de
instrucciones GCN (idéntico); simulador de consola con el costo de GPU de cada
kernel medido con el código máquina real (`tools/isa_costs.py` →
`plugin/test/kernel_costs.h`): con el juego usando 24 ms de GPU, la calidad dinámica
cambia Rápido por Ligero y alcanza, sin atrasos.

## Novedades de la 2.73

Con la 2.72 en GTA V la generación de VCFG anduvo perfecto; la IA llegaba a ~40
fps y el reescalado, el SMAA y la nitidez no se notaban (o no estaban activos).

**Generación IA nueva: NXFG-2** (`model/net2.py`, `model/bm2.py`, `shaders/nxflow.cl`).
La red anterior (NXFlow) apenas le ganaba al block-matching y costaba 1,17 GMAC.
Lo que más pesa en la calidad es el movimiento, así que ahí va la GPU:

- **Movimiento de 4 niveles a media resolución** (luma 960x544 → 120x68),
  bloques de 8x8, radios 4/3/3/2, mediana 3x3 por nivel, **ajuste sub-píxel** y
  **confianza** por bloque. Alcance de ~200 px entre dos fotogramas reales (la
  anterior: ~40 px; en GTA manejando rápido no alcanzaba). Malla de 1/16, como VCFG.
- **Red 5 veces más liviana** (~0,23 GMAC a 1080p): nivel A a 1/16 (32 canales,
  interno 1/32) que corrige los dos flujos y estima la oclusión con la confianza
  como entrada, y nivel B a 1/8 (24 canales) para los bordes.
- **Warp por bloques superpuestos con elección por píxel**: cada píxel mezcla
  el warp de las 4 celdas vecinas con su peso bilineal (OBMC: sin bordes de
  bloque) multiplicado por lo bien que cada vector hace coincidir los dos
  fotogramas reales (softmin): en los bordes de objetos gana el vector del lado
  correcto, sin halos. Es la idea del warp de VCFG con sus vecinos. La red se
  entrena con este warp (JAX), no es un agregado aparte.
- **Protecciones fijas que no dependen de la red** (anti-parpadeo):
  - *píxel quieto*: si los dos reales coinciden en ese píxel (HUD, textos,
    zonas sin cambio) se usa su promedio sin warp. El HUD queda firme;
  - *fantasmas de HUD*: con el fondo moviéndose rápido, el warp copiaba el HUD
    corrido (copias medio transparentes arriba y abajo, en VCFG y en la 2.72
    también). Ahora una muestra que cae sobre un píxel quieto (máscara de media
    resolución, dilatada) con un vector que se mueve no se usa: se toma la del
    otro fotograma;
  - *corte de escena*: si la confianza media del movimiento es mala, el
    intermedio repite el real en vez de inventar una mezcla.
- Un fotograma que no tenía anterior válido dejaba la pirámide de un fotograma
  viejo para el siguiente (un intermedio malo suelto = un parpadeo): corregido.
- Camino de GPU verificado contra la referencia NumPy en el emulador: IA y sin
  red, ~94 dB y 1/255 de diferencia máxima.
- **Medido en PC** (`model/eval2.py`, detalle en `docs/resultados.md` §1b):

  | escenas tipo juego (HUD, oclusiones) | PSNR | SSIM | error en HUD/zonas quietas |
  |---|---|---|---|
  | block-matching tipo VCFG | 24,44 | 0,791 | 2,00 |
  | NXFlow (IA de la 2.72) | 24,60 | 0,796 | 2,07 |
  | **NXFG-2 IA (2.73)** | **24,75** | **0,815** | **0,18** |

  En video real queda ~0,15 dB por debajo del block-matching (mismo SSIM). Gana
  justo donde estaban los parpadeos y fantasmas, con 5 veces menos cálculo.

**Reescalado, SMAA y nitidez funcionando, también con la generación.**

- Hasta la 2.72 se apagaban con cualquier generación activa (filas grises en el
  menú) y con VCFG ni se reservaba su memoria. Ahora el real se procesa en un
  buffer propio y vuelve al del juego **antes** de generar: el intermedio sale de
  reales ya procesados y hereda la misma calidad (una sola pasada por fotograma
  del juego, sin diferencia de nitidez entre real e intermedio = sin "pulso").
- **NXSR más visible:** a la resolución nativa es realce de detalle (no hay nada
  que escalar) y con ganancia 1 cambiaba ~0,7/255 en promedio. Suave / Normal /
  Fuerte = 1 / 2 / 3 (el limitador 3x3 del shader sigue impidiendo halos).
- **FSR+** ("Nitidez FSR+"): RCAS (al máximo de FSR desde Media) y después un
  realce de detalle de radio 2 sobre la luma que recupera lo que suavizan TAA y
  FXAA, sin halos (no sale del rango 3x3 local) y sin subir el grano. Nitidez
  medida (energía de gradiente) Baja/Media/Alta: ×1,43 / ×1,70 / ×2,06 contra
  ×1,34 / ×1,52 / ×1,91 del RCAS anterior.
- **El menú dice qué está activo**: "activo: VCFG+NXSR+SMAA+FSR" (o IA, EASU,
  "-C" para el modelo Calidad, y "carga: ... (-n)" si la calidad dinámica bajó
  algo). Si falta memoria lo dice con los MiB que faltan.
- **Comparar** muestra una línea divisoria: izquierda original, derecha procesada.
- **Calidad dinámica por tiempo de GPU** con la generación: si nuestro trabajo
  pasa de 8 ms por fotograma (le quitaría GPU al juego), se baja primero el
  modelo Calidad, después SMAA, después NXSR/nitidez; la generación queda.
- **Menos memoria:** la memoria de trabajo de los efectos a 1080p baja de 62,7 a
  42,6 MiB (pirámides a media resolución, red chica), y además no se reservan
  intermedios propios cuando el juego no usa flips de CPU (todos usan la
  presentación de VCFG) y queda un solo buffer para el real procesado sin EASU
  (25 MiB menos). En total, unos 45 MiB menos que la 2.72 en GTA V.
- Simulador: escenario 30 (VCFG + NXSR + SMAA + FSR+: memoria, flips, trabajo);
  emulador: modo 7 (FSR+) contra `upscale/fsr1.py`.

**Emulador de la GPU de la PS4 a nivel de instrucciones** (`tools/gcn_isa_emu.py`).
Un emulador de PS4 completo (shadPS4 y similares) no sirve acá: no carga plugins
de GoldHEN, necesita el juego y una GPU, y su GPU es una traducción a Vulkan, no
la de la consola. En cambio se emula lo que importa: el **código máquina GCN
gfx701** que el plugin copia a la memoria de la GPU, instrucción por instrucción
(wavefronts de 64 lanes, EXEC/VCC/SCC, SGPR/VGPR, LDS, descriptores T#/S#/V#
leídos de memoria, control de rango de los buffers), ejecutando el mismo DCB PM4
que arma el plugin.

- LLVM 18 no desensambla gfx7, así que los `.s` se ensamblan con `llvm-mc` y se
  verifica que los bytes sean **idénticos** a los de `nx_shaders.h` y a los de la
  memoria en cada `COMPUTE_PGM` (31 de 31 kernels) antes de ejecutarlos.
- `plugin/test/test_isa_path.py` corre los 8 modos del pipeline (IA, sin red,
  RCAS, FSR+, SMAA, NXSR Rápido y Calidad, IA sobre reales con NXSR): el código
  máquina da lo mismo que `gpu_emu` (diferencia máxima 2/255; casi siempre 0),
  sin accesos fuera de la memoria reservada, sin instrucciones que la PS4 no tenga
  y sin bucles que no terminen. Lo que no modela: tiempos, el tiling de los
  buffers del juego y la precisión de 8 bits del filtro bilineal del hardware.

**Ideas de VCFG medidas** (el análisis de todas las versiones: las v2.40–v2.54 que
había eran nuestras; de VCFG, v2.14, v2.23 y el código v2.25). Se probaron sobre
NXFG-2 con PSNR y con el error temporal de la secuencia mostrada (real,
intermedio, real...) contra la verdadera; ninguna mejoró, así que no se adoptan:

| idea de VCFG | val | video | error temporal (menos es mejor) |
|---|---|---|---|
| NXFG-2 tal cual | 30,73 | 37,65 | 0,789 |
| guard de píxel quieto 3x más estricto | 30,54 | 37,58 | 0,813 |
| muestra fuera de pantalla → la del otro lado | 30,53 | 37,65 | 0,789 |
| camino coherente (un solo vector si difieren ≤1 px) | 30,65 | 37,62 | 0,791 |
| softmin más suave (τ 0,15 + piso 0,05) | 30,73 | 37,65 | 0,789 |
| predictor temporal en el block-matching | — | 37,43–37,50 | 0,790–0,793 |

De VCFG sí se tomó lo que ya estaba probado bien: la presentación (intermedios,
colas, pacing) y la idea de elegir vector por píxel entre los vecinos. Su
post-proceso in situ no se copió: VCFG lo documenta como fuente de parpadeo (el
warp leía el real procesado con la pirámide del crudo, y había carrera entre
grupos); acá el post es fuera de lugar, se copia después de una barrera y la
pirámide se arma del real ya procesado.

## Novedades de la 2.72

Con la 2.71 en GTA V la generación de VCFG anduvo bien (30 → 60); la IA propia
seguía bajando los fps y el reescalado no se notaba.

- **Sin escalera de arranque:** siempre el nivel completo. Solo si el juego se
  cerró con los efectos recién encendidos ("pendfx"), el arranque siguiente baja
  una parte; el que sale bien vuelve arriba. Salir mientras carga no cuenta.
- **Generación IA con la presentación de VCFG.** Lo que funcionó con VCFG ahora
  lo usa también nuestra generación: 3 intermedios con el formato del juego,
  4 trabajos en vuelo (la CPU casi nunca espera), prólogo que espera al juego en
  la GPU, `WaitFlipDone` sobre el intermedio, flip al intermedio + flip del real
  al buffer del juego, juego a 30 con salida a 60 Hz, menú en los dos fotogramas.
  Entre el prólogo y el final corre la cadena propia (pirámides, block-matching,
  NXFlow y warp) escribiendo directo en el intermedio.
- **Block-matching con LDS (`wg_bm_search`):** un grupo de 64 hilos por bloque
  carga una vez las dos zonas de luma ((8+2R)² cada una) y evalúa los candidatos
  desde ahí: ~16 veces menos lecturas de memoria (el block-matching leía ~270
  millones de valores por fotograma a 1080p). Resultado idéntico bit a bit al de
  antes (emulador, modos IA, rápida y con nitidez).
- **Reescalado más visible:** NXSR Suave/Normal/Fuerte = 0,75 / 1,1 / 1,5 (antes
  0,5 / 0,75 / 1,0; es la ganancia del residuo, el modelo no cambia) y la nitidez
  RCAS se puede usar junto con NXSR (antes se apagaba). Las etapas intermedias del
  post-proceso alternan buffers.
- La cadena propia tiene 4 slots de parámetros (pirámides por paridad).
- Simulador: escenario 29 (generación IA con la presentación de VCFG); los de la
  calidad dinámica corren sin generación.

## Novedades de la 2.71

**Generación de fotogramas de VCFG (motor clásico) portada al plugin.** Con la
2.69/2.70 en GTA V (PS4 Slim) la generación propia funcionaba (salida a 60 Hz)
pero costaba ~62 ms de GPU por fotograma: el juego caía a 12 fps. VCFG
(github.com/Melnikoff07/vcfg, licencia MIT) lleva GTA V de 30 a 60 en una PS4
base con ~3-4 ms. Su código es abierto, así que la generación se porta tal cual
(crédito y licencia en `plugin/vcfg8/`):

- **Shaders de VCFG** (`pyr8`, `me3`, `me2`, `gm8`, `fgs8`, `me1`, `mf8`,
  `phase8`, `warp8`), compilados desde sus fuentes `.ll` con su propio
  `gen_shaders.py` → `plugin/include/vcfg8_shaders.h`.
- **`nx_fg8.c`**: portado de `build_dcb_flow8` / `fg8_table.h` / `fg_res.h`.
  Pirámide de luma fija (960x540, 480x270, 240x135; solo la del fotograma
  actual, la anterior por paridad), estimación de movimiento de grueso a fino,
  fase y warp a resolución completa. Prólogo y epílogo de VCFG: el CP (no la CPU)
  espera el fin del trabajo del juego, `ACQUIRE_MEM` de color/L2/K$, y
  `sceGnmInsertWaitFlipDone` sobre el intermedio justo antes del warp.
- **Presentación como VCFG:** un envío con los DCB del juego (sin su cola) + el
  nuestro (con la cola copiada) y flip al **intermedio**; después un envío con la
  cola sola y flip al **buffer del juego**, con el mismo `flipMode` y `flipArg`
  del juego. **3 intermedios** con el formato, tiling y pitch del juego,
  rotando; 4 slots de trabajo (la CPU casi nunca espera).
- **El juego queda a 30 parejo** con la salida forzada a 60 Hz (como el
  `fps_lock 30` de VCFG): entre envíos pasan al menos 33 ms.
- **Menú y contador sin parpadeo:** se dibujan en el intermedio y en el real en
  el mismo trabajo, y el warp toma el fotograma actual dentro del rectángulo del
  menú (protección de overlay de `warp8`).
- Menú: **Generación de fotogramas: OFF / VCFG / IA.** VCFG es la de por defecto;
  con ella el reescalado, el SMAA y la nitidez no se aplican (como en VCFG) y no
  se reserva su memoria. "IA" es la generación con NXFlow (pesada).
- Simulador: escenario 28 (flip intermedio + real por fotograma, colas, 60 Hz,
  pausa con la GPU trabada, sin memoria no se activa). Los escenarios de la
  calidad dinámica usan la generación IA.

## Novedades de la 2.70

Con la 2.69 en GTA V (PS4 Slim) la generación Rápida funcionó con la salida a
60 Hz, pero costaba **61,8 ms de GPU** por fotograma (el juego bajó a 12 fps) y
los valores del menú parpadeaban.

- **Generación Rápida ~4 veces más liviana.** El block-matching comparaba cada
  bloque de 8x8 con 49 posiciones en las tres escalas (incluida la de 1080p):
  ~200 millones de lecturas por fotograma. Ahora, en Rápida, busca con radio 4
  solo en la escala chica y refina ±1 en las otras dos (rango ±19 px), y no
  calcula los promedios RGB que solo usa la red. La generación con IA no cambia
  (radio 3: es con lo que se entrenó). En el emulador, con desplazamientos de
  8 a 32 px en diagonal, Rápida queda a la par de IA.
- **Sin parpadeo del menú con la generación.** El intermedio sale de deformar
  fotogramas que traen el menú del anterior grabado; ahora bajo cada panel del
  intermedio se copia la imagen real limpia antes de dibujarlo.
- Cambiar entre Rápida e IA rehace la pirámide de luma.

## Novedades de la 2.69

**La salida a 60 Hz ya no se prende y apaga en cada fotograma.** Para que la
generación sirva, el plugin fuerza `sceVideoOutSetFlipRate(0)` (GTA V pide 1:
30 Hz). Antes se soltaba en cualquier fotograma sin generación (por ejemplo tras
un atraso de la GPU) y se volvía a forzar en el siguiente: la salida saltaba
entre 30 y 60 Hz. Ahora queda forzada mientras la generación esté en uso y se
suelta recién tras ~2 s seguidos sin ella. Cada cambio queda en el registro.

El menú muestra la tasa de salida en la primera línea ("fps · GPU x ms · 60Hz").

## Novedades de la 2.68

GTA V corre con efectos sin colgarse (2.67, niveles 4-6 probados en consola).
Pero la generación con la red NXFlow a 1080p bajaba el juego a ~10 fps: GTA V
ya usa casi toda la GPU.

- **Generación de fotogramas: OFF / Rápida / IA.** "Rápida" usa los vectores del
  block-matching (escala 1/8 → 1/4) y el warp, sin las capas de NXFlow (19
  despachos en lugar de 44); es el tipo de generación de VCFG. "IA" es la de
  siempre. Los modelos no cambian. En el emulador, con un desplazamiento de
  8 px: Rápida 41,2 dB, IA 42,8 dB, mezcla simple 19,0 dB. `framegen = 1` en el
  .ini es Rápida, `2` es IA.
- **Tiempo de GPU propio en el menú** ("GPU x ms") y en el registro: dos marcas
  del reloj de la GPU (`EVENT_WRITE_EOP` con DATA_SEL 3) al empezar y terminar el
  trabajo propio; la unidad se calibra contra el reloj de la CPU.
- Con una versión nueva se conserva el nivel de la escalera ya alcanzado (si se
  había cerrado, una parte menos).
- `gpu_emu`: modo 6 (generación rápida).

## Novedades de la 2.67

Con la 2.66, GTA V en nivel 4: los efectos anduvieron ~600 fotogramas en la
pantalla de carga ("efectos: arranque correcto") y el juego se colgó al empezar
la partida.

- **El motor ya no toca estado global de compute.** El pipeline escribía
  `COMPUTE_TMPRING_SIZE = 0`, `STATIC_THREAD_MGMT_SE0/1` y `RESOURCE_LIMITS`; ese
  estado queda en el anillo para lo que el juego envía después, y los shaders
  del juego que usan scratch encontraban un anillo de scratch de tamaño 0. VCFG
  no los toca y nuestros kernels no usan scratch.
- **La imagen real se presenta en el buffer del juego**, como VCFG: el resultado
  procesado se copia de vuelta al buffer del juego (`copy_back`) y el flip real
  va al índice del juego; el intermedio de la generación va al buffer propio.
  Así el juego sigue viendo sus buffers en pantalla. (Con EASU, salida más
  grande que el juego, se sigue presentando el buffer propio.)
- **Efectos por partes:** nivel 4 = nitidez (RCAS), 5 = + reescalado IA y SMAA,
  6 = + generación de fotogramas. Cada arranque correcto suma una parte; si una
  se cuelga, la próxima vez se usa la anterior. Con una versión nueva, lo que
  llegó al menú o más arranca en la parte 1.
- **Hitos de los efectos en la traza, al instante:** memoria lista, primer
  trabajo (con qué efectos), primer fotograma generado, GPU atrasada.
- Simulador: escenario 27 (las tres partes; el último flip va a un buffer del
  juego).

## Novedades de la 2.66

**El menú se centra bien sin efectos reservados.** En GTA V (nivel 3) el menú
aparecía arriba a la izquierda con 3/4 fuera de la pantalla: se centraba con el
tamaño de salida (`Wo`/`Ho`), que solo se calcula al reservar los efectos y
valía 0. Ahora el OSD usa el tamaño de la imagen donde se dibuja (la del juego
si no hay efectos) y nunca una posición negativa. Simulador: el escenario 26
verifica que el menú quede dentro de la pantalla.

## Novedades de la 2.65

**La escalera de arranque ya no se traba en "2 de 2".** En el nivel 3 (solo
menú) el arranque contaba como correcto recién con ~600 trabajos propios de GPU,
pero en ese nivel solo hay trabajo con el menú abierto: salir del juego quedaba
como "se cerró" y el tope bajaba a 2. Ahora:

- el arranque cuenta como correcto a los 900 fotogramas del juego, en cualquier
  nivel;
- en el nivel 4, cuando se encienden los efectos la marca vuelve a "pendiente"
  ("efectos encendidos: a prueba") hasta ~600 fotogramas con efectos;
- con una versión nueva del plugin el tope vuelve a 4 y el registro anterior
  cuenta como correcto (0-2 → 3, 3 → 4); Diagnóstico muestra lo mismo.

## Novedades de la 2.64

**El menú L3+R3 ya se dibuja en GTA V (2.63, confirmado en consola)**, pero se
veía solo la parte de arriba. Al final de cada trabajo propio el DCB escribía el
label de fin pero no lo esperaba: el flip que va detrás mostraba la imagen con el
compute a medio hacer. Ahora, como VCFG, después del `EVENT_WRITE_EOP` final el
CP espera ese label (`WAIT_REG_MEM`) e invalida caches (`ACQUIRE_MEM`) antes de
la cola de flip.

En el nivel 3 (solo menú) la línea de estado del menú dice "solo menú (efectos:
próx. arranque)" y las opciones de efectos se ven atenuadas.

## Novedades de la 2.63

Con la 2.62 GTA V ya corre a 30 fps con el plugin; se colgaba justo cuando el
motor mandaba su primer trabajo de GPU ("menú listo" → "carga alta" → cuelgue).

- **Shaders sin instrucciones FLAT.** Compilados para gfx7 con `amdhsa`, LLVM
  usaba `flat_load/flat_store` para la memoria global; FLAT depende de las
  aperturas que prepara el driver de HSA (`SH_MEM_BASES`), que el de la PS4 no
  configura. VCFG solo usa instrucciones de buffer e imagen. Ahora
  `shaders/build_gcn.sh` compila con `-flat-for-global` desactivado: el backend
  usa MUBUF `addr64` (descriptor de base 0 y dirección de 64 bits en VGPRs, como
  radeonsi en SI/CI) y el script falla si queda alguna instrucción FLAT. Los
  modelos no cambian.
- **Sincronización como VCFG.** Antes de leer la imagen, el DCB propio espera a
  que el juego termine de dibujarla (`EVENT_WRITE_EOP` BOTTOM_OF_PIPE + 
  `WAIT_REG_MEM` + `ACQUIRE_MEM` sobre el buffer); entre pasadas,
  `ACQUIRE_MEM` en lugar de `SURFACE_SYNC`.
- **Escalera de arranque con el riesgo de GPU separado:** nivel 3 = solo el menú
  (un shader chico), nivel 4 = efectos + memoria temprana. Tras un arranque
  correcto en 0-2 se salta al 3; del 3 al 4. El registro de versiones
  anteriores a la 2.63 no se usa (el motor cambió): se empieza de 0.
- Simulador: escenario 26 (nivel 3: solo menú).

## Novedades de la 2.62

**Guarda contra el anidamiento de los hooks de flip.** Con la 2.61, GTA V en
nivel 4 mostró "carga alta" y el aviso de los 25 s marcó **~30000 fps**; la
partida nunca terminó de cargar y el sistema cerró el juego.
`sceGnmSubmitAndFlipCommandBuffers` llama por dentro a
`sceGnmSubmitAndFlipCommandBuffersForWorkload` (VCFG lo registra como
"nested in ours" y lo deja pasar con `g_in_submit`). VCFG-NX engancha las dos:
el motor volvía a procesar su propio envío desde adentro, en cadena.

Ahora los tres hooks de presentación tienen una guarda: lo que llega mientras ya
se está dentro de un envío va directo a la función original, sin motor ni
conteo.

## Novedades de la 2.61

**El trabajo propio va en el mismo envío que el del juego, con la "cola de
flip" (como VCFG).** Con la 2.59 GTA V llegó al nivel 3 ("efectos activos"),
avisó "carga alta" y al abrir el menú se colgó. `sceGnmSubmitAndFlipCommandBuffers`
exige que el último DCB termine en el paquete de `prepareFlip` (NOP de 64 dwords
con la marca `0x687507xx`; GTA V usa `0x68750778`), que el sistema reescribe al
presentar. VCFG-NX enviaba el trabajo del juego aparte y su DCB sin esa cola:
el sistema escribía sobre las últimas instrucciones del DCB propio (el label de
fin de trabajo no llegaba → "carga alta"; con el menú, la GPU colgada).

Ahora, igual que el motor clásico de VCFG:

- un solo envío con flip: los DCB del juego (el último sin sus 64 dwords) + el
  DCB propio, que termina con una copia de la cola;
- el flip extra de la generación de fotogramas usa otra copia de la cola;
- si el último DCB del juego no termina en la cola, el fotograma pasa intacto
  (`sin_cola_de_flip` en el registro, con los dwords finales);
- PS4 Pro: los T# de buffers con tiling 2D llevan `alt_tile_mode` (bit 24 de
  `t[6]`), como libSceGnm (sin él, franjas verticales);
- con efectos, el arranque cuenta como correcto recién cuando el trabajo propio
  lleva ~600 fotogramas (un cierre al encenderlos baja el nivel).

Simulador: el DCB del juego lleva la cola real; cada flip se verifica que
termine en ella; escenarios 25a (marca de GTA V) y 25b (sin cola: intacto).

## Novedades de la 2.60

**GTA V arranca con el plugin (confirmado en consola con la 2.59).** La escalera
de arranque ahora sube directo:

- después de un arranque correcto (900 fotogramas) el próximo va **directo al
  nivel completo** (menú L3+R3 y efectos); si ese se cierra, baja de a uno;
- se aprovecha el registro de la 2.59 (desde ahí la pila se realinea): un
  arranque correcto con la 2.59 ya cuenta;
- un cierre en el nivel 0 (que no toca nada) ya no apaga el plugin para ese
  juego: se vuelve a probar el nivel 0.

## Novedades de la 2.59

**Causa del cierre de GTA V encontrada: la pila llega desalineada.** La traza de
la 2.58 se cortó justo después de "plugin_load: inicio". La instrucción
siguiente en el binario era un `movaps` sobre la pila (el compilador inicializa
así el título del juego con `-O2`). GoldHEN llama a `plugin_load` desde el parche
de `_init_env` de la libc del juego, y en GTA V la pila llega alineada a 8 y no
a 16: `movaps` provoca una excepción y el juego se cierra. Por eso VCFG compila
su motor clásico sin optimizar ("-O2 crashed CUSA00207").

Arreglo: el plugin se compila con `-mstackrealign`: cada función realinea su
pila al entrar (`and $-16, %rsp`), sin perder la optimización.

## Novedades de la 2.58

**Ningún hilo propio dentro de `plugin_load`.** Con la 2.57 GTA V mostró el
aviso de `module_start` y después se cerró. En VCFG (código fuente) ningún hilo
se crea en `plugin_load`: el de E/S, el del mando y el monitor arrancan desde los
flips, con el juego ya andando. VCFG-NX creaba el hilo "vcfg_nx_boot" en
`plugin_load` (antes también el del mando). Ahora:

- no hay hilo de arranque: el "arranque correcto" se marca desde los flips del
  juego (a los 900 fotogramas);
- nivel 0 = aviso + los tres flips enganchados **solo para contarlos** (pasan
  derecho al sistema); niveles 1-2 también cuentan sin procesar; desde el
  nivel 3 el motor procesa los flips;
- el hilo del mando sigue arrancando a los 300 fotogramas y solo desde el nivel 3.

## Novedades de la 2.57

**Sin stdio dentro del arranque del juego.** GoldHEN carga los plugins desde
`_init_env` de la libc del juego, antes de que el juego termine de iniciarla.
El código fuente de VCFG (github.com/Melnikoff07/vcfg, `classic/source/boot_stage.h`)
lo documenta: *"stdio inside plugin_load faults in libSceLibcInternal"*. La traza
de la 2.55/2.56 usaba `fopen` desde `module_start` y cerraba GTA V apenas
arrancaba (por eso no salía ningún aviso). Ahora:

- la traza se escribe con `sceKernelOpen/Write/Close` (abrir, agregar, cerrar,
  0666), igual que los registros de arranque de VCFG, en
  `/data/GoldHEN/vcfg_nx/traza.txt` y en `/data/vcfg_nx/traza.txt`;
- `final_printf` ya no llama a `printf`: el plugin no importa ninguna función
  de stdio (ni `fopen`, `fputs`, `printf`, `mkdir` de libc);
- los registros se abren sin `O_FSYNC` y con 0666, como VCFG.

## Novedades de la 2.56

**La respuesta sale en pantalla, no solo en archivos.** En GTA V la traza tenía
una sola línea ("module_start") y no se podía saber si venía del juego o de la
prueba de carga de la propia app. Ahora:

- la app carga el plugin para probarlo con el argumento `vcfgapp`, y la traza
  lo anota como "prueba de la app"; Diagnóstico solo cuenta las líneas del juego;
- dentro del juego, `module_start` muestra el aviso "VCFG-NX: GoldHEN abrió el
  plugin" apenas se abre el plugin, antes de `plugin_load`;
- el aviso de `plugin_load` dice si se pudo escribir el registro (o el código de
  error) y si funcionó la traza;
- en el nivel 0 sin registro posible, el plugin no crea ni el hilo de arranque:
  solo el aviso;
- se corrigió la última línea de la traza en Diagnóstico (mostraba un carácter
  de más).

## Novedades de la 2.55

**Registros dentro de la carpeta de GoldHEN (`/data/GoldHEN/vcfg_nx`).** En
GTA V el diagnóstico mostraba que el plugin nunca había dejado ningún registro,
ni siquiera el del arranque escalonado, así que ese sistema no podía guardar
nada entre intentos. Ahora:

- los registros (`cargas.txt`, `boot-<ID>.txt`, `<ID>.log`, `guard-<ID>.txt`)
  se escriben en `/data/GoldHEN/vcfg_nx`, la misma carpeta donde GoldHEN ya
  escribe sin problemas;
- hay una segunda vía, `traza.txt`, escrita con `fopen` de libc (como la
  v2.23) desde `module_start`: si una vía falla, la otra muestra hasta dónde
  llegó el plugin;
- Diagnóstico muestra la última línea de la traza y cuántas hay; el reporte
  incluye el final de `traza.txt`; la app sigue leyendo los registros viejos
  de `/data/vcfg_nx/logs` si no hay nuevos;
- "Limpiar todo y reinstalar" también borra la carpeta nueva; los permisos
  quedan en 0777.

Los ajustes (`config.ini`, `<ID>.ini`) siguen en `/data/vcfg_nx`.

## Novedades de la 2.54

**Arranque ascendente.** Cada juego empieza con el plugin sin tocar nada y,
por cada arranque que llega a jugarse (40 s, o 900 fotogramas desde el nivel
3), suma una parte: 0 solo aviso → 1 configuración y motor → 2 registro de
pantalla → 3 presentación (menú L3+R3 y efectos) → 4 memoria temprana. Si un
nivel cierra el juego, el siguiente arranque vuelve al último que funcionó y
ese queda como tope: a lo sumo un cierre, y el nivel donde pasó señala la
parte culpable. El Diagnóstico muestra el próximo nivel y el tope; "Limpiar
todo" vuelve a empezar.

## Novedades de la 2.53

**Enganche propio en lugar de `HOOK32` del SDK de GoldHEN.** `HOOK32` pone el
puente hacia el plugin en memoria de `malloc` y salta ahí con un `jmp` de 32
bits (±2 GB): si esa memoria queda lejos de la función enganchada, el salto
cae en cualquier lado y el juego se cierra la primera vez que llama a la
función (al arrancar). Que funcione depende de cómo quede la memoria de cada
plugin en cada juego. `nx_hook` (plugin/source/nx_hook.c):

- pide la memoria del puente **ejecutable y cerca** de la función y verifica
  la distancia antes de usarla;
- decodifica (HDE64) las instrucciones que copia y **corrige** las relativas
  (RIP-relativas, `call`/`jmp` rel32); si hay una que no se puede corregir,
  **no engancha** esa función (queda intacta) en vez de dejar un salto roto;
- sigue los thunks (`jmp` rel32/corto/`[rip]`) hasta la función real;
- anota el resultado de cada enganche en `logs/cargas.txt`.

Probado en el PC con funciones x86-64 reales (`make -C plugin/test hook`):
prólogo común, lectura RIP-relativa, `call` al principio, thunk y rechazo de
un salto condicional.

## Novedades de la 2.52

Desensamblando los dos motores de VCFG v2.23 (`vcfg16` y `vcfg_classic42`,
el que usa con GTA V): enganchan las mismas funciones dentro de
`plugin_load`, pero **el hilo que lee el mando lo arrancan recién cuando el
juego ya está presentando imagen**. El nuestro arrancaba en `plugin_load` y
llamaba a UserService/Pad mientras GTA V todavía se inicializaba. Ahora el
hilo se crea a los 300 fotogramas del juego (~10 s), desde el hook de flip.

Además: la marca de arranque y las etapas se escriben con `O_FSYNC`
(sobreviven a un apagado forzado, así el nivel prudente se aplica de verdad)
y cada versión nueva vuelve a empezar desde el nivel 0.

## Novedades de la 2.51

**Causa del cierre de GTA V al arrancar.** El plugin buscaba funciones del
sistema por nombre (`sceKernelGetModuleList` + `sceKernelDlsym` sobre cada
módulo del juego) dentro de `plugin_load`, mientras el juego todavía se
inicializa. En la 2.48 eso corría justo después del aviso (el juego se cerraba
apenas salía la notificación); en la 2.50 se movió antes del aviso (para
`sceKernelFsync`) y el cierre pasó a ser antes de la notificación: el cierre
se movió con esa función. Se quitó por completo (sin control de temperatura
ni fsync). La marca de arranque "pendiente" ahora es lo primero que se
escribe, así el nivel prudente se aplica pase lo que pase después.

## Novedades de la 2.50

Cuando el juego se cerraba la consola quedaba colgada (no se podía cerrar el
juego ni apagar). Para que cada prueba falle lo menos posible y deje rastro:

- **El plugin no usa la GPU hasta que el arranque quedó confirmado**: el
  arranque por defecto pasa de 600 a 1200 fotogramas (~40 s) y la marca de
  "arranque correcto" se pone a los 900. Si algo falla al iniciar, no puede
  ser trabajo nuestro en la GPU.
- Nuestros comandos de GPU no tienen esperas del lado de la GPU (solo
  escrituras de fin de trabajo): no pueden dejar la cola de la GPU trabada.
- **Las marcas de arranque y las etapas se escriben al disco al momento**
  (`sceKernelFsync`, buscado por nombre): sobreviven a un apagado forzado, y
  así el nivel de arranque más prudente se aplica en el siguiente intento.

## Novedades de la 2.49

Con la 2.48 el plugin ya carga (aparece el aviso), pero GTA V se cerraba al
arrancar. Cambios:

- **Sin la reserva de 112 MiB al arrancar** (principal sospechosa: GTA V se
  queda sin memoria). Solo se reservan 4 MiB para el menú; los efectos piden
  memoria más tarde y solo si queda margen.
- **Arranque por niveles** (como el crash guard de v2.23): si un arranque no
  llega a jugarse (~30 s de fotogramas), el siguiente usa un nivel más
  prudente — 0 normal, 1 sin reserva temprana, 2 pasivo (sin hooks) — y el
  que funciona queda guardado para ese juego. Aviso en pantalla con el nivel.
- **Etapas anotadas al momento** en `logs/cargas.txt` (inicio, memoria,
  hooks, primer fotograma, pantalla registrada, menú, efectos, jugando); el
  Diagnóstico muestra hasta dónde llegó el último arranque de cada juego.
- La sonda se retira (ya cumplió) y sin el hook de
  `sceKernelAllocateMainDirectMemory`.

## Novedades de la 2.48

**Conflictos fuera, automáticamente.** Al guardar (o sea, apenas se abre la
app y después de cada cambio), en cada juego con VCFG-NX activado y en
`[default]` se quitan de `plugins.ini` los plugins que tocan lo mismo:
GNMScale, versiones viejas de VCFG (`vcfg16`, `vcfg_classic42`, `bb_probe`),
`fliprate_remover`, `force_30_fps`, `force_1080p_display`, `frame_logger` y
otros escaladores/generadores de fotogramas. Los demás plugins (parches,
`afr`, etc.) quedan. Hay copia de seguridad de `plugins.ini` y la app muestra
qué quitó.

## Novedades de la 2.47

**Error 0x8002000D (EACCES, permiso denegado) al cargar el plugin.** El
Diagnóstico de una consola real mostró que la sonda y el plugin daban
`0x8002000D` al cargarse. El cargador de GoldHEN hace `chmod 0777` antes de
cargar cada plugin, pero desde el proceso del juego no puede cambiarle los
permisos a un archivo que creó otra app. Ahora la app deja 0777 el plugin, la
sonda, `plugins.ini`, `/data/GoldHEN/plugins` y `/data/vcfg_nx` (y sus
archivos) cada vez que se abre. El Diagnóstico muestra los permisos de
`vcfg_nx.prx` junto a los de `gnmscale_plugin.prx` y prueba cargar también el
de GNMScale como control.

## Novedades de la 2.46

**OPTIONS → Limpiar todo y reinstalar.** Borra todo lo que dejaron versiones
anteriores de VCFG-NX: `/data/vcfg_nx` entera (ajustes, registros,
protecciones, marcas de arranque), cualquier `vcfg_nx*` en
`/data/GoldHEN/plugins`, archivos temporales `.vcfg_tmp`, copias viejas de
`plugins.ini` y las líneas de VCFG-NX en `plugins.ini` (antes hace una copia
nueva; los demás plugins no se tocan). Después instala esta versión de cero en
los juegos activados, con los ajustes por defecto. Incluye la corrección de
`exp2f` de la 2.45.

## Novedades de la 2.45

**Causa encontrada de que el plugin no cargara en ningún juego.** Comparando
cada función que importa el plugin con las que importan VCFG v2.23, GNMScale
y el plugin de ejemplo de GoldHEN, la única distinta era `exp2f` (de
`libSceLibcInternal`, la usaba la nitidez RCAS). Si la consola no la exporta,
la PS4 no puede enlazar el PRX y GoldHEN no lo carga: el juego arranca como
si nada, sin aviso ni menú. Desde la 1.0 estaba. Ahora se calcula dentro del
plugin (mismo resultado, error < 1e-6). Plugin, sonda y app importan solo
funciones que también usan esos binarios que funcionan.

Revisión de compatibilidad de los modelos en la GPU de PS4: ningún shader usa
memoria scratch; LDS máx. 23 KB (NXSR Calidad; límite 32 KB); VGPR máx. 81/256;
SGPR máx. 94/104.

## Novedades de la 2.44

- **Sin hooks del mando**, como VCFG v2.23 (`pad entry hooks disabled`): el
  menú se maneja solo con el hilo que lee el mando por su cuenta.
- El aviso "VCFG-NX cargado en CUSA…" sale **lo primero** al cargar (antes
  salía al final de plugin_load) y el estado en pantalla a los 25 s.
- **Prueba de carga en la app** (Diagnóstico): la app abre la sonda y el
  plugin instalados con `sceKernelLoadStartModule`, como GoldHEN, y muestra
  "OK" o el código de error exacto de la consola.

## Novedades de la 2.43

**Juegos que ocupan toda la memoria (GTA V).** Como VCFG v2.23, el plugin
reserva su memoria apenas GoldHEN lo carga, antes de que el juego arranque:
4 MiB para el menú (siempre) y 112 MiB para los efectos. Si el juego después
pide memoria y no hay, se le devuelve la de los efectos (mientras no se haya
usado) y su pedido se repite: el juego nunca se queda sin memoria por el
plugin. Antes se esperaba ~20 s para pedirla y GTA V ya no dejaba ni para el
menú, por eso L3+R3 no hacía nada. Si un arranque con reserva no llega a
jugarse, el siguiente arranca sin la reserva de efectos.

**Sonda de diagnóstico.** La app instala también `vcfg_nx_probe.prx` (mínimo,
solo usa el kernel) antes del plugin en la sección del juego. Al abrir el
juego avisa "GoldHEN cargó plugins en CUSA…" y 12 s después si el plugin
principal quedó cargado. A los 40 s el plugin muestra su estado (resolución,
fps, o por qué no actúa).

## Novedades de la 2.42

**Guardado automático.** Hasta la 2.41 la app solo instalaba el plugin y
escribía `plugins.ini` al salir con CÍRCULO → "Guardar y salir". Si se cerraba
con el botón PS (lo habitual), los juegos se veían activados pero en disco no
había nada: GoldHEN no cargaba VCFG-NX en ningún juego. Ahora, como VCFG v2.23,
cada cambio se guarda solo medio segundo después, y al abrir la app se
instala/actualiza el plugin enseguida.

## Novedades de la 2.41

Diagnóstico para encontrar en qué punto se corta en cada consola: la app
(OPTIONS → Diagnóstico, con el juego elegido) revisa si el plugin está
instalado y es la misma versión, si `plugins.ini` tiene la sección del juego,
si hay **otros plugins en ese juego** que pueden chocar (GNMScale, otro VCFG),
si GoldHEN llegó a cargar VCFG-NX alguna vez y qué dejó el plugin en el
registro del juego. Guarda todo en `/data/vcfg_nx/reporte.txt`. El plugin
deja una marca en `logs/cargas.txt` en el instante en que GoldHEN lo abre.

## Novedades de la 2.4

**Funcionar en los juegos.** En GTA V y Marvel's Spider-Man el plugin no hacía
nada y L3+R3 no respondía. Comparando con VCFG v2.23 (que sí anda) se
encontraron y corrigieron estas causas posibles:

- **Una importación que el juego no resuelve.** La 2.3 importaba
  `sceKernelGetCpuTemperature`, que solo exportan algunos módulos del sistema:
  si el proceso del juego no la tiene, GoldHEN no puede cargar el plugin y el
  juego corre como si nada. Ahora esa función (y las lecturas `Ext` del mando)
  se buscan por nombre en los módulos cargados; si no están, se sigue sin ellas.
- **Archivos con la libc.** Ahora se usan las llamadas del kernel
  (`sceKernelOpen/Read/Write/Mkdir`), como v2.23.
- **El mando.** Antes L3+R3 dependía de enganchar `scePadReadState/scePadRead`;
  muchos juegos leen por `scePadReadStateExt/ReadExt` u otra vía. Ahora un hilo
  del plugin lee el mando por su cuenta cada 16 ms (como v2.23) y los hooks
  quedan solo para que el juego no reciba botones con el menú abierto. Si el
  menú todavía no puede abrirse, una notificación dice por qué.
- **Registro de pantallas más tolerante.** Antes había que registrar 2 a 8
  buffers de una sola vez en ciertos formatos; si no, el plugin quedaba
  esperando para siempre (y con él el menú). Ahora acepta buffers registrados
  de a uno, en cualquier índice, ignora otras salidas, y un formato raro ya no
  lo apaga si después llega la pantalla real.
- **Aviso al cargar**: al abrir el juego aparece "VCFG-NX cargado. Mantener
  L3+R3 para el menu". Si no aparece, GoldHEN no cargó el plugin en ese juego.
- **Registro por juego** en `/data/vcfg_nx/logs/` y su resumen en Diagnóstico.

**Interfaz nueva "Aurora"**, distinta de GNMScale: fondo grafito con luces
turquesa y violeta, paneles de vidrio esmerilado de verdad (lo de atrás se ve
desenfocado), bordes que brillan arriba, portadas sin marco con anillo de luz
al elegir, filtros en un control de vidrio centrado, tipografía Inter Display
liviana, icono nuevo y el menú del juego con el mismo estilo.

## Novedades de la 2.3

![lista de juegos](docs/img/app_juegos.jpg)

* **Interfaz rediseñada**: fuente Inter con suavizado, tarjetas con bordes
  redondeados, sombras, brillo en la tarjeta elegida, paneles translúcidos,
  interruptores y selectores, íconos de los botones del mando.
* **Sonidos** al moverse entre juegos, elegir, volver, cambiar de pestaña,
  guardar y en los errores (generados por código, en un hilo propio).
* **Arreglado "Error al guardar"**: la app escapaba del sandbox (para leer la
  lista de juegos) antes de copiar el plugin desde `/app0`, y afuera esa ruta
  ya no es la app. Ahora el plugin se lee al arrancar, con rutas de respaldo,
  y si algo falla se muestra el motivo exacto con opciones (reintentar, ver
  diagnóstico, salir sin guardar). Sin plugin no se toca ningún archivo.
* **Íconos**: se buscan en varias carpetas del sistema (`appmeta`) y, si no
  están, se sacan del `app.pkg` instalado de cada juego (la entrada
  `icon0.png` del PKG no está cifrada; también el nombre, de `param.sfo`).
  Se cargan de a uno mientras no se toca el mando. **Opciones → Diagnóstico** muestra cuántos
  cargaron y, si alguno falla, la ruta.
* **Salir** abre un diálogo: guardar y salir, salir sin guardar o cancelar.
* **Fluidez**: solo se redibuja la zona que cambia (moverse entre juegos
  redibuja dos tarjetas y el panel); sombras y marcos se calculan una vez.

![ajustes de un juego](docs/img/app_ajustes_juego.jpg)

## Novedades de la 2.2


* **App nueva**: cuadrícula con los juegos instalados y sus íconos, filtros
  (todos / activos / con ajustes propios) y un panel con lo que va a usar el
  juego elegido.
* **Ajustes por juego** desde la app: cada juego puede seguir los ajustes por
  defecto o tener los suyos (FG, NXSR, modelo, SMAA, RCAS, calidad dinámica,
  modo ahorro, límite de temperatura, contador de FPS). Es el mismo archivo
  que guarda el menú L3+R3 dentro del juego.
* **Menos memoria**: los buffers temporales del FG y del post-proceso
  comparten la misma región: 90 → 63 MiB a 1080p (−31 %), con salida idéntica
  byte a byte. Más juegos tienen memoria libre para activar los efectos.
* **La app no usa CPU en reposo**: solo redibuja cuando se toca un botón (antes
  redibujaba la pantalla entera 30 veces por segundo).
* Juegos en discos externos (`ext0`/`ext1`) sin repetir. Los modelos (NXFlow y
  NXSR) no cambiaron.

## Novedades de la 2.1

* **Dos modelos NXSR**: Rápido (16 canales, ~1 000 operaciones por píxel) y
  Calidad (24 canales y dos capas, ~3 300), que en "Auto" se usa solo si el
  juego corre en modo PS4 Pro. Calidad además limpia ruido en zonas planas y
  deja menos halos que la imagen original.
* **Calidad dinámica**: si la GPU no da abasto baja un escalón (Calidad →
  Rápido → sin SMAA → sin NXSR → sin FG) en vez de apagarse, y vuelve a subir
  sola cuando hay margen.
* **Protección térmica**: lee la temperatura cada 2 s; sobre 80 °C baja la
  calidad, sobre 86 °C pausa los efectos hasta que enfríe. **Modo ahorro**:
  modelo Rápido, sin SMAA y límite de 75 °C.
* **Más estable**: si el juego vuelve a registrar sus buffers (al volver del
  reposo, por ejemplo) se adapta en lugar de apagarse.
* **Más eficiente**: NXSR lee cada píxel una vez en vez de tres; el modelo
  Rápido se reentrenó con el doble de datos y una pérdida que reduce el
  parpadeo en movimiento (−14 %).

## Protecciones (para que no se cierre el juego ni pase nada a la consola)

* **Nada al arrancar**: el plugin no reserva memoria ni toca la GPU hasta que
  el juego lleva ~20 s mostrando imagen (`warmup_flips`).
* **Memoria**: solo reserva si al juego le quedan libres al menos `reserve_mb`
  (192 MiB) más lo necesario; si no, los efectos quedan apagados y reintenta
  más tarde. El menú necesita solo 32 MiB.
* **Anti-cierre**: si un juego se cierra con los efectos activos, la próxima
  vez arranca en **modo seguro** (efectos apagados, se avisa por notificación).
  Tras ~1 minuto estable, la sesión se marca como buena.
* **GPU**: si nuestro trabajo no termina a tiempo, ese fotograma sale intacto;
  si pasa 30 veces seguidas, el plugin se desactiva solo para esa sesión.
* **Cualquier cosa rara** (HDR, otro tamaño de buffers, índice desconocido,
  proceso que no es un juego) → el fotograma del juego pasa sin tocar.
* El instalador hace copia de seguridad de `plugins.ini`, verifica cada
  escritura y restaura el sandbox de la app al salir. Nunca escribe fuera de
  `/data/GoldHEN` y `/data/vcfg_nx`.

## Compatibilidad

Funciona con cualquier juego (no hay lista fija): se elige en la app. Soporta
las tres formas en que los juegos de PS4 presentan la imagen
(`sceGnmSubmitAndFlipCommandBuffers`, la variante `ForWorkload` y
`sceVideoOutSubmitFlip` desde la CPU), salida SDR de 8 y 10 bits de 720p a 4K
(PS4 Pro), y juegos que renderizan por debajo de la salida (900p/720p → 1080p
con EASU). Si el juego ya va a más de `max_game_fps` (48), la generación se
apaga sola y solo queda la mejora de imagen. Con salida HDR solo se ofrece el
menú (los efectos se calibraron en SDR). No se toca: apps que no son juegos
(`CUSA`/`PLAS`).

**Si un juego no responde**, abrir la app → OPTIONS → **Diagnóstico** con ese
juego elegido: resume lo que el plugin dejó en `/data/vcfg_nx/logs/<ID>.log`
(si cargó, qué formato de pantalla usa el juego, cuántos fotogramas procesó y
por qué pasó intactos los demás). `/data/vcfg_nx/logs/cargas.txt` lista cada
vez que el plugin cargó en un juego. Ese archivo `.log` es lo que hay que
compartir para arreglar un juego puntual.

## Instalar y usar

1. Requiere **GoldHEN** con soporte de plugins.
2. Instalar `VCFG-NX_v2.54.pkg` y abrir **VCFG-NX**.
3. En la cuadrícula: **X** activa o desactiva VCFG-NX en el juego,
   **CUADRADO** abre los ajustes de ese juego, **TRIÁNGULO** los ajustes por
   defecto, **L1/R1** cambia el filtro, **OPTIONS** abre el menú (activar o
   desactivar todos, quitar ajustes propios, diagnóstico, desinstalar) y
   **CÍRCULO** sale. Todo se guarda solo al cambiarlo.
   En los ajustes de un juego, cambiar cualquier valor lo pasa a "ajustes
   propios"; **CUADRADO** lo devuelve a los ajustes por defecto.
4. En el juego: mantener **L3+R3** 1,5 s abre el menú (el juego no recibe los
   botones mientras está abierto). Ahí se cambia cada efecto en vivo, se ve el
   contador de FPS, el modo **Comparar** (mitad izquierda original) y
   **Guardar** deja los ajustes para ese juego.

Configuración (`/data/vcfg_nx/config.ini`, o `/data/vcfg_nx/<TITLE_ID>.ini` por juego):

```ini
effects = 1         ; interruptor general (menu L3+R3)
framegen = 1        ; 30 -> 60
sr = 2              ; NXSR: 0 off, 1 suave, 2 normal, 3 fuerte
sr_model = 0        ; 0 auto (Calidad en PS4 Pro), 1 rapido, 2 calidad
dynamic = 1         ; calidad dinamica
eco = 0             ; modo ahorro (menos calor)
temp_limit = 80     ; grados C; 0 = sin control termico
aa = 1              ; SMAA 1x
rcas = 0            ; nitidez FSR1 cuando NXSR esta apagado: 0..3
fps_counter = 0
compare = 0         ; mitad izquierda sin procesar
max_game_fps = 48   ; por encima, FG se apaga solo
menu_hold_ms = 1500 ; L3+R3 mantenido -> menu
warmup_flips = 600  ; fotogramas antes de reservar nada
reserve_mb = 192    ; memoria que siempre se le deja libre al juego
```

## Contenido

```
dist/                          PKG, plugin .prx y sumas SHA256
docs/ANALISIS_VCFG_v2.14.md    qué hay dentro del VCFG original
docs/ANALISIS_GNMSCALE.md      qué hay dentro de GNMScale y por qué agrega grano
docs/DISENO_NX.md              arquitectura, estados, protecciones, ABI de los shaders
docs/resultados.md             métricas y verificación
model/                         NXFlow (frame generation): datos, red, entrenamiento, export
sr/                            NXSR: datos, red, entrenamiento, evaluación (con GNMScale), export
upscale/                       referencias FSR1 y SMAA
shaders/                       kernels de GPU (.cl) -> ISA GCN gfx701
plugin/source/                 plugin: motor (nx_engine), tubería GPU, menú en pantalla, hooks
plugin/test/                   host_test, emulador del DCB, simulador de consola con ASan
installer/                     app (cuadrícula de juegos, ajustes por juego), edición segura de plugins.ini, PKG
installer/test/                test_core, test_app (la app entera con mando guionado) y capturas de pantalla
weights/                       pesos entrenados (nxflow.pkl, nxsr.pkl, nxsr_hq.pkl)
```

## Compilar y probar

```sh
pip install "jax[cpu]" optax pillow pyelftools pyyaml
shaders/build_gcn.sh                                   # .cl -> ISA gfx701 -> plugin/include/nx_shaders.h
make -C plugin/test all
plugin/test/host_test                                  # ABI de kernargs + PM4 de cada caso
plugin/test/console_sim                                # 18 escenarios: memoria, cierres, GPU, calor, menú
python3 plugin/test/test_gpu_path.py video.mp4 weights/nxflow.pkl
python3 plugin/test/test_smaa_path.py
python3 plugin/test/test_sr_path.py                    # NXSR: DCB emulado vs modelo
make -C installer/test                                 # app: archivos, interfaz de punta a punta
make -C installer OO_PS4_TOOLCHAIN=... GOLDHEN_SDK=... [SSL11_DIR=carpeta con libssl.so.1.1]
```

Reentrenar NXSR (con capturas del juego mejora más):

```sh
python3 sr/data_sr.py --images carpeta/ --videos captura.mp4 --out train_sr.npz
python3 sr/train_sr.py --data train_sr.npz --val val_sr.npz --C 16 --body 1 --margin 0 --lr 8e-3 --batch 32 --steps 12000 --out weights/nxsr.pkl
python3 sr/eval_sr.py --data val_sr.npz --nxsr weights/nxsr.pkl --gnmscale carpeta_models_gnmscale/
python3 sr/export_sr.py weights/nxsr.pkl plugin/include/nxsr_weights.h && shaders/build_gcn.sh
```

## ¿Y DLSS (HelixSR)?

No se puede usar en PS4: es una DLL de Windows/Direct3D 12, necesita vectores
de movimiento que los juegos de PS4 no exponen, apunta a GPUs RDNA, su licencia
prohíbe modificarla y los pesos de NVIDIA no se pueden redistribuir. Por eso
se agregó SMAA 1x (1.0) y ahora NXSR, entrenado desde cero.

## Créditos y licencias

* VCFG y GNMScale: autores de la comunidad GoldHEN. Este repo no incluye sus
  binarios ni sus pesos; solo el análisis.
* SMAA: Jimenez et al., MIT (`third_party/smaa/LICENSE.txt`). FSR1: AMD, MIT.
* Fuente del menú: DejaVu Sans Mono (licencia libre de Bitstream Vera/DejaVu).
* stb_image (Sean Barrett): dominio público / MIT, para los íconos de la app.
* Fuente de la app: Inter (The Inter Project Authors), SIL Open Font License 1.1
  (`third_party/inter/OFL.txt`); los glifos pre-renderizados siguen bajo la misma licencia.
* Formato PKG/PFS: LibOrbisPkg (maxton). Toolchain: OpenOrbis y GoldHEN Plugins SDK.
* Datos: Big Buck Bunny (© Blender Foundation, CC-BY 3.0), Urban100, BSD100,
  Set14, Sun-Hays80, clips de muestra de scikit-video/OpenCV y escenas
  sintéticas propias.
