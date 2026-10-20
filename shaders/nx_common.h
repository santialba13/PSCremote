// Utilidades comunes de los kernels VCFG-NX.
//
// No se usa ningun builtin de OpenCL que dependa de las device-libs de ROCm
// (get_global_id, read_imagef, exp, clamp...): con clang "pelado" quedan como
// llamadas sin resolver. En su lugar:
//   * ids con __builtin_amdgcn_* y grupos fijos de 8x8 (reqd_work_group_size);
//   * matematica con __builtin_* que el backend AMDGPU baja a instrucciones;
//   * imagenes con descriptores T#/S# crudos (shaders/gcn_image.ll).
#pragma once

#define NX_WG 8
#ifdef NX_HOST
// compilacion para x86 (plugin/test/gpu_emu.c): el emulador recorre la grilla.
// Los kernels "wg_*" se llaman una vez por grupo y recorren sus 64 hilos con
// WG_BEGIN/WG_END en cada fase (entre fases va WG_BARRIER).
int nx_host_gid_x(void);
int nx_host_gid_y(void);
int nx_host_group_x(void);
int nx_host_group_y(void);
#define NX_KERNEL __kernel
#define GID_X nx_host_gid_x()
#define GID_Y nx_host_gid_y()
#define GROUP_X nx_host_group_x()
#define GROUP_Y nx_host_group_y()
#define WG_BEGIN for (int lid_ = 0; lid_ < NX_WG * NX_WG; ++lid_) {
#define WG_END }
#define LID lid_
#define WG_BARRIER
#else
#define NX_KERNEL __kernel __attribute__((reqd_work_group_size(NX_WG, NX_WG, 1)))
#define GID_X ((int)(__builtin_amdgcn_workgroup_id_x() * NX_WG + __builtin_amdgcn_workitem_id_x()))
#define GID_Y ((int)(__builtin_amdgcn_workgroup_id_y() * NX_WG + __builtin_amdgcn_workitem_id_y()))
#define GROUP_X ((int)__builtin_amdgcn_workgroup_id_x())
#define GROUP_Y ((int)__builtin_amdgcn_workgroup_id_y())
#define WG_BEGIN {
#define WG_END }
#define LID ((int)(__builtin_amdgcn_workitem_id_y() * NX_WG + __builtin_amdgcn_workitem_id_x()))
#define WG_BARRIER { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); \
                     __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); }
#endif

typedef __constant const uint* tdesc_t;   // T# (8 dwords) o S# (4 dwords)

float4 nx_load(tdesc_t t, int x, int y);
float4 nx_sample(tdesc_t t, tdesc_t s, float x, float y);
void nx_store(tdesc_t t, int x, int y, float4 v);

static inline float nx_minf(float a, float b) { return __builtin_fminf(a, b); }
static inline float nx_maxf(float a, float b) { return __builtin_fmaxf(a, b); }
static inline float nx_absf(float a) { return __builtin_fabsf(a); }
static inline float nx_clampf(float v, float lo, float hi) { return nx_minf(nx_maxf(v, lo), hi); }
static inline int nx_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float nx_floor(float a) { return __builtin_floorf(a); }
static inline float nx_rcp(float a) { return 1.0f / a; }
static inline float nx_rsqrt(float a) { return 1.0f / __builtin_sqrtf(a); }
static inline float nx_sigmoid(float a) { return 1.0f / (1.0f + __builtin_exp2f(-1.44269504f * a)); }

static inline float4 nx_min4(float4 a, float4 b)
{ return (float4)(nx_minf(a.x, b.x), nx_minf(a.y, b.y), nx_minf(a.z, b.z), nx_minf(a.w, b.w)); }
static inline float4 nx_max4(float4 a, float4 b)
{ return (float4)(nx_maxf(a.x, b.x), nx_maxf(a.y, b.y), nx_maxf(a.z, b.z), nx_maxf(a.w, b.w)); }
static inline float4 nx_clamp4(float4 v, float4 lo, float4 hi) { return nx_min4(nx_max4(v, lo), hi); }
