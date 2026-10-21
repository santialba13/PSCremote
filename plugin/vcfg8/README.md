# Generación de fotogramas del motor clásico de VCFG

Shaders y lógica de la generación de fotogramas ("mode 8") del motor clásico de
VCFG: github.com/Melnikoff07/vcfg (commit 63a7b2f), licencia MIT
(Copyright (c) 2026 VCFG contributors, ver `LICENSE`).

- `*.ll`: fuentes LLVM IR gfx700 de `classic/shaders/`, sin cambios.
- `gen_shaders.py`: el generador de VCFG (`classic/gen_shaders.py`), sin cambios.
- `plugin/include/vcfg8_shaders.h`: generado con
  `python3 plugin/vcfg8/gen_shaders.py llc plugin/vcfg8 plugin/include/vcfg8_shaders.h`.
- `plugin/source/nx_fg8.c`: portado de `classic/source/main.c` (`build_dcb_flow8`)
  y `classic/source/fg8_table.h` / `fg_res.h`.
