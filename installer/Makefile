# Instalador de VCFG-NX y su PKG (fPKG para PS4 con GoldHEN).
#   make OO_PS4_TOOLCHAIN=/ruta/OpenOrbis/PS4Toolchain GOLDHEN_SDK=/ruta/GoldHEN_Plugins_SDK
# Compila tambien el plugin (../plugin) y lo mete en el PKG en /app0/plugins/.
TITLE       := VCFG-NX
VERSION     := 2.54
TITLE_ID    := VCNX00001
CONTENT_ID  := IV0000-VCNX00001_00-VCFGNXINSTALL000

TOOLCHAIN := $(OO_PS4_TOOLCHAIN)
BIN       := $(TOOLCHAIN)/bin/linux
# PkgTool.Core es .NET Core 3: necesita libssl 1.1 (en distros nuevas, pasar su carpeta en SSL11_DIR)
PKGTOOL   := LD_LIBRARY_PATH=$(SSL11_DIR) DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 $(BIN)/PkgTool.Core
SAMPLE    := $(TOOLCHAIN)/samples/hello_world
OUT       := build
STAGE     := $(OUT)/stage

CFLAGS  := --target=x86_64-pc-freebsd12-elf -fPIC -funwind-tables -O2 -Wall \
           -isysroot $(TOOLCHAIN) -isystem $(TOOLCHAIN)/include -I$(GOLDHEN_SDK)/include
LDFLAGS := -m elf_x86_64 -pie --script $(TOOLCHAIN)/link.x --eh-frame-hdr -L$(TOOLCHAIN)/lib -L$(GOLDHEN_SDK) \
           -lGoldHEN_Hook -lc -lkernel -lSceSystemService -lSceVideoOut -lSceAudioOut -lScePad -lSceUserService -lSceSysmodule \
           $(TOOLCHAIN)/lib/crt1.o

all: $(OUT)/$(CONTENT_ID).pkg

../plugin/probe/build/vcfg_nx_probe.prx: FORCE
	$(MAKE) -C ../plugin/probe OO_PS4_TOOLCHAIN=$(TOOLCHAIN) GOLDHEN_SDK=$(GOLDHEN_SDK)

../plugin/build/vcfg_nx.prx: FORCE
	$(MAKE) -C ../plugin

$(OUT)/%.o: source/%.c source/inst_core.h source/ui_gfx.h source/ui_sound.h source/ui_font_data.h ../plugin/source/config.h
	@mkdir -p $(OUT)
	clang $(CFLAGS) -c -o $@ $<

# lectura/escritura de los .ini: el mismo codigo que usa el plugin
$(OUT)/config.o: ../plugin/source/config.c ../plugin/source/config.h
	@mkdir -p $(OUT)
	clang $(CFLAGS) -c -o $@ $<

OBJS := $(OUT)/main.o $(OUT)/inst_core.o $(OUT)/ui_gfx.o $(OUT)/ui_sound.o $(OUT)/config.o

$(STAGE)/eboot.bin: $(OBJS)
	@mkdir -p $(STAGE)
	ld.lld $(OBJS) -o $(OUT)/installer.elf $(LDFLAGS)
	$(BIN)/create-fself -in=$(OUT)/installer.elf -out=$(OUT)/installer.oelf --eboot $@ --paid 0x3800000000000011

$(STAGE)/sce_sys/param.sfo: Makefile
	@mkdir -p $(STAGE)/sce_sys
	rm -f $@
	$(PKGTOOL) sfo_new $@
	$(PKGTOOL) sfo_setentry $@ APP_TYPE --type Integer --maxsize 4 --value 1
	$(PKGTOOL) sfo_setentry $@ APP_VER --type Utf8 --maxsize 8 --value '$(VERSION)'
	$(PKGTOOL) sfo_setentry $@ ATTRIBUTE --type Integer --maxsize 4 --value 0
	$(PKGTOOL) sfo_setentry $@ CATEGORY --type Utf8 --maxsize 4 --value 'gd'
	$(PKGTOOL) sfo_setentry $@ CONTENT_ID --type Utf8 --maxsize 48 --value '$(CONTENT_ID)'
	$(PKGTOOL) sfo_setentry $@ DOWNLOAD_DATA_SIZE --type Integer --maxsize 4 --value 0
	$(PKGTOOL) sfo_setentry $@ SYSTEM_VER --type Integer --maxsize 4 --value 0
	$(PKGTOOL) sfo_setentry $@ TITLE --type Utf8 --maxsize 128 --value '$(TITLE)'
	$(PKGTOOL) sfo_setentry $@ TITLE_ID --type Utf8 --maxsize 12 --value '$(TITLE_ID)'
	$(PKGTOOL) sfo_setentry $@ VERSION --type Utf8 --maxsize 8 --value '$(VERSION)'

$(STAGE)/sce_sys/icon0.png: make_icon.py
	@mkdir -p $(STAGE)/sce_sys
	python3 make_icon.py $@

stage: $(STAGE)/eboot.bin $(STAGE)/sce_sys/param.sfo $(STAGE)/sce_sys/icon0.png ../plugin/build/vcfg_nx.prx \
       ../plugin/probe/build/vcfg_nx_probe.prx
	mkdir -p $(STAGE)/sce_sys/about $(STAGE)/sce_module $(STAGE)/plugins
	cp $(SAMPLE)/sce_sys/about/right.sprx $(STAGE)/sce_sys/about/
	cp $(SAMPLE)/sce_module/libc.prx $(SAMPLE)/sce_module/libSceFios2.prx $(STAGE)/sce_module/
	cp ../plugin/build/vcfg_nx.prx ../plugin/probe/build/vcfg_nx_probe.prx $(STAGE)/plugins/

$(OUT)/$(CONTENT_ID).pkg: stage
	cd $(STAGE) && $(BIN)/create-gp4 -out pkg.gp4 --content-id=$(CONTENT_ID) --files \
	    "eboot.bin sce_sys/about/right.sprx sce_sys/param.sfo sce_sys/icon0.png sce_module/libc.prx sce_module/libSceFios2.prx plugins/vcfg_nx.prx plugins/vcfg_nx_probe.prx"
	# create-gp4 solo declara las carpetas estandar: agregar plugins/
	sed -i 's|<dir targ_name="sce_module" />|<dir targ_name="sce_module" />\n\t\t<dir targ_name="plugins" />|' $(STAGE)/pkg.gp4
	cd $(STAGE) && $(PKGTOOL) pkg_build pkg.gp4 ..
	@ls -la $@

clean:
	rm -rf $(OUT)

.PHONY: all stage clean FORCE
