#pragma once
static inline int sceUserServiceInitialize(void* p) { return 0; }
static inline int sceUserServiceGetInitialUser(int* u) { *u = 1; return 0; }
