#!/bin/sh
# Compila los kernels a ISA GCN gfx701 (generacion de la GPU de PS4) y los
# empaqueta en plugin/include/nx_shaders.h.
# Requiere clang/llvm-link/ld.lld >= 17 con el target AMDGPU (el de Ubuntu sirve).
set -eu
cd "$(dirname "$0")"
OUT=${OUT:-build}
mkdir -p "$OUT"
# -flat-for-global: en gfx7 con amdhsa, LLVM usa instrucciones FLAT para la
# memoria global, y FLAT depende de las aperturas que configura el driver de
# HSA (SH_MEM_BASES), que el de la PS4 no prepara: una direccion mal resuelta
# cuelga la GPU. Sin FLAT, el backend usa MUBUF addr64 (descriptor con base 0
# y la direccion de 64 bits en VGPRs), como radeonsi en SI/CI. VCFG tampoco usa
# FLAT (solo buffer/imagen).
MATTR="-Xclang -target-feature -Xclang -flat-for-global"
CLFLAGS="-target amdgcn-amd-amdhsa -mcpu=gfx701 $MATTR -nogpulib -O3 -cl-std=CL1.2"

# NXSR: canales y capas los fija cada modelo exportado (Rapido y Calidad)
hdr() { sed -n "s/^#define $1 \\([0-9]*\\)/\\1/p" "../plugin/include/$2"; }
SR_FLAGS="-DSR_C=$(hdr NXSR_C nxsr_weights.h) -DSR_B=$(hdr NXSR_B nxsr_weights.h) -DSR_NAME=wg_nxsr"
HQ_FLAGS="-DSR_C=$(hdr NXSR_HQ_C nxsr_hq_weights.h) -DSR_B=$(hdr NXSR_HQ_B nxsr_hq_weights.h) -DSR_NAME=wg_nxsr_hq"
clang $CLFLAGS -emit-llvm -c gcn_image.ll -o "$OUT/gcn_image.bc"
for k in nxflow fsr1 smaa nxsr nxsr_hq osd nxsrl nxre; do
    case $k in
        nxsr) src=nxsr.cl; flags=$SR_FLAGS ;;
        nxsr_hq) src=nxsr.cl; flags=$HQ_FLAGS ;;
        *) src=$k.cl; flags= ;;
    esac
    clang $CLFLAGS -I. $flags -emit-llvm -c "$src" -o "$OUT/$k.bc"
    llvm-link "$OUT/$k.bc" "$OUT/gcn_image.bc" -o "$OUT/$k.linked.bc"
    clang -target amdgcn-amd-amdhsa -mcpu=gfx701 $MATTR -nogpulib -O3 "$OUT/$k.linked.bc" -o "$OUT/$k.co"
    clang -target amdgcn-amd-amdhsa -mcpu=gfx701 $MATTR -nogpulib -O3 -S "$OUT/$k.linked.bc" -o "$OUT/$k.s"
    if grep -qE "^\s+flat_" "$OUT/$k.s"; then echo "ERROR: $k usa instrucciones FLAT" >&2; exit 1; fi
done
python3 ../tools/gcn_pack.py "$OUT/nxflow.co" "$OUT/fsr1.co" "$OUT/smaa.co" "$OUT/nxsr.co" "$OUT/nxsr_hq.co" "$OUT/osd.co" \
    "$OUT/nxsrl.co" "$OUT/nxre.co" \
    --asm "$OUT/nxflow.s" "$OUT/fsr1.s" "$OUT/smaa.s" "$OUT/nxsr.s" "$OUT/nxsr_hq.s" "$OUT/osd.s" "$OUT/nxsrl.s" "$OUT/nxre.s" \
    -o ../plugin/include/nx_shaders.h
