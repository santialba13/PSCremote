#pragma once
#include <stdint.h>
struct jailbreak_backup { int x; };
#include <stdio.h>
// fake_hide_app0: como en la consola, fuera del sandbox /app0 ya no es la app
static int fake_hide_app0;
static inline int sys_sdk_jailbreak(struct jailbreak_backup* j)
{
#ifdef ROOT
    if (fake_hide_app0) rename(ROOT "/app0", ROOT "/app0_sandbox");
#endif
    return 0;
}
static inline int sys_sdk_unjailbreak(struct jailbreak_backup* j)
{
#ifdef ROOT
    if (fake_hide_app0) rename(ROOT "/app0_sandbox", ROOT "/app0");
#endif
    return 0;
}
static inline unsigned sys_sdk_version(void) { return 0x0204; }
