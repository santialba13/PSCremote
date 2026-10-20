#pragma once
#include <setjmp.h>
// "exit": en el test vuelve al llamador con longjmp
static jmp_buf fake_exit;
static int fake_exit_armed;
static inline int sceSystemServiceLoadExec(const char* p, const char* a[]) { if (fake_exit_armed) longjmp(fake_exit, 1); return 0; }
