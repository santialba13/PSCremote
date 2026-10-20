; Acceso a imagenes con descriptores crudos de GCN (T# de 8 dwords, S# de 4).
; Es lo mismo que hace un shader de GNM: el plugin construye el T# a partir
; del render target del juego (formato, tiling, pitch) y lo pasa en memoria.
; Asi los kernels leen el backbuffer con el tiling que use el juego sin
; convertirlo a lineal. Se enlaza con los .cl mediante llvm-link.

target triple = "amdgcn-amd-amdhsa"

declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32)

; float4 nx_load(__constant uint* t, int x, int y)   -- image_load, sin filtrado
define <4 x float> @nx_load(ptr addrspace(4) %t, i32 %x, i32 %y) #0 {
  %rsrc = load <8 x i32>, ptr addrspace(4) %t, align 32
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; float4 nx_sample(__constant uint* t, __constant uint* s, float x, float y)
; coordenadas en pixeles (unnormalized), filtro segun el S# (bilineal en el plugin)
define <4 x float> @nx_sample(ptr addrspace(4) %t, ptr addrspace(4) %s, float %x, float %y) #0 {
  %rsrc = load <8 x i32>, ptr addrspace(4) %t, align 32
  %samp = load <4 x i32>, ptr addrspace(4) %s, align 16
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 15, float %x, float %y, <8 x i32> %rsrc, <4 x i32> %samp, i1 true, i32 0, i32 0)
  ret <4 x float> %v
}

; void nx_store(__constant uint* t, int x, int y, float4 v)
define void @nx_store(ptr addrspace(4) %t, i32 %x, i32 %y, <4 x float> %v) #1 {
  %rsrc = load <8 x i32>, ptr addrspace(4) %t, align 32
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %v, i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

attributes #0 = { alwaysinline nounwind readonly }
attributes #1 = { alwaysinline nounwind }
