# Compila vcfg_nx.prx (plugin de GoldHEN).
#   make OO_PS4_TOOLCHAIN=/ruta/OpenOrbis/PS4Toolchain GOLDHEN_SDK=/ruta/GoldHEN_Plugins_SDK
# GOLDHEN_SDK tiene que estar compilado antes (make en su carpeta: libGoldHEN_Hook.a y build/crtprx.o).
TOOLCHAIN := $(OO_PS4_TOOLCHAIN)
GH_SDK    := $(GOLDHEN_SDK)
OUT       := build
TARGET    := $(OUT)/vcfg_nx.prx

LIBS := -lGoldHEN_Hook -lSceLibcInternal -lkernel -lSceSysmodule -lSceVideoOut -lSceGnmDriver -lScePad -lSceUserService

# -mstackrealign: GoldHEN llama a plugin_load desde el parche de _init_env de la
# libc del juego y en algunos juegos (GTA V) la pila llega desalineada (8 en vez
# de 16): cualquier movaps sobre la pila cerraba el juego ahi mismo. Cada funcion
# realinea su pila al entrar.
CFLAGS  := --target=x86_64-pc-freebsd12-elf -fPIC -funwind-tables -O2 -mstackrealign -Wall -Wno-unused-parameter \
           -isysroot $(TOOLCHAIN) -isystem $(TOOLCHAIN)/include -I$(GH_SDK)/include -Iinclude -Isource -I../third_party/smaa
LDFLAGS := -m elf_x86_64 -pie --script $(TOOLCHAIN)/link.x -e _init --eh-frame-hdr \
           -L$(TOOLCHAIN)/lib -L$(GH_SDK) $(LIBS)

SRCS := $(wildcard source/*.c)
OBJS := $(patsubst source/%.c,$(OUT)/%.o,$(SRCS))

all: $(TARGET)

$(OUT)/%.o: source/%.c include/nx_shaders.h
	@mkdir -p $(OUT)
	clang $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	ld.lld $(GH_SDK)/build/crtprx.o $(OBJS) -o $(OUT)/vcfg_nx.elf $(LDFLAGS)
	$(TOOLCHAIN)/bin/linux/create-fself -in=$(OUT)/vcfg_nx.elf -out=$(OUT)/vcfg_nx.oelf \
	    --lib=$@ --paid 0x3800000000000011

clean:
	rm -rf $(OUT)

.PHONY: all clean
