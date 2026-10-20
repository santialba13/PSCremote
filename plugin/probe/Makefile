# Sonda de diagnostico (vcfg_nx_probe.prx): solo libkernel y el SDK de GoldHEN.
#   make OO_PS4_TOOLCHAIN=... GOLDHEN_SDK=...
TOOLCHAIN := $(OO_PS4_TOOLCHAIN)
GH_SDK    := $(GOLDHEN_SDK)
OUT       := build
CFLAGS  := --target=x86_64-pc-freebsd12-elf -fPIC -funwind-tables -O2 -Wall -Wno-unused-parameter \
           -isysroot $(TOOLCHAIN) -isystem $(TOOLCHAIN)/include -I$(GH_SDK)/include
LDFLAGS := -m elf_x86_64 -pie --script $(TOOLCHAIN)/link.x -e _init --eh-frame-hdr \
           -L$(TOOLCHAIN)/lib -L$(GH_SDK) -lGoldHEN_Hook -lSceLibcInternal -lkernel

$(OUT)/vcfg_nx_probe.prx: source/probe.c
	@mkdir -p $(OUT)
	clang $(CFLAGS) -c -o $(OUT)/probe.o $<
	ld.lld $(GH_SDK)/build/crtprx.o $(OUT)/probe.o -o $(OUT)/vcfg_nx_probe.elf $(LDFLAGS)
	$(TOOLCHAIN)/bin/linux/create-fself -in=$(OUT)/vcfg_nx_probe.elf -out=$(OUT)/vcfg_nx_probe.oelf \
	    --lib=$@ --paid 0x3800000000000011

clean:
	rm -rf $(OUT)
