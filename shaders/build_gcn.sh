#!/bin/sh
# Compila los kernels a ISA GCN gfx701 (generacion de la GPU de PS4) y los
# empaqueta en plugin/include/nx_shaders.h.
# Requiere clang/llvm-link/ld.lld >= 17 con el target AMDGPU (el de Ubuntu sirve).
set -eu
cd "$(dirname "$0")"
OUT=${OUT:-build}
mkdir -p "$OUT"
CLFLAGS="-target amdgcn-amd-amdhsa -mcpu=gfx701 -nogpulib -O3 -cl-std=CL1.2"

clang $CLFLAGS -emit-llvm -c gcn_image.ll -o "$OUT/gcn_image.bc"
for k in nxflow fsr1 smaa; do
    clang $CLFLAGS -I. -emit-llvm -c "$k.cl" -o "$OUT/$k.bc"
    llvm-link "$OUT/$k.bc" "$OUT/gcn_image.bc" -o "$OUT/$k.linked.bc"
    clang -target amdgcn-amd-amdhsa -mcpu=gfx701 -nogpulib -O3 "$OUT/$k.linked.bc" -o "$OUT/$k.co"
    clang -target amdgcn-amd-amdhsa -mcpu=gfx701 -nogpulib -O3 -S "$OUT/$k.linked.bc" -o "$OUT/$k.s"
done
python3 ../tools/gcn_pack.py "$OUT/nxflow.co" "$OUT/fsr1.co" "$OUT/smaa.co" \
    --asm "$OUT/nxflow.s" "$OUT/fsr1.s" "$OUT/smaa.s" \
    -o ../plugin/include/nx_shaders.h
