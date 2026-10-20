# Tests de la parte portable del plugin en el PC.
#   make run        -> host_test: ABI de kernargs + PM4 bien formado
#   make emu        -> gpu_emu: ejecuta el DCB con los kernels .cl compilados para x86
#   python3 plugin/test/test_gpu_path.py   -> compara gpu_emu con el modelo de referencia
CC = clang
CFLAGS ?= -O2 -Wall -Wextra
PORTABLE = ../source/pm4.c ../source/gcn_desc.c ../source/nx_pipeline.c ../source/nx_dispatch.c ../source/nx_osd.c
HEADERS = ../include/nx_shaders.h ../include/nxflow_weights.h ../include/nxsr_weights.h

all: host_test gpu_emu console_sim

host_test: host_test.c $(PORTABLE) $(HEADERS)
	$(CC) $(CFLAGS) -I../source -o $@ host_test.c $(PORTABLE) -lm

kernels_host.o: ../../shaders/nxflow.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

fsr1_host.o: ../../shaders/fsr1.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

smaa_host.o: ../../shaders/smaa.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

SR_C = $(shell sed -n 's/^\#define NXSR_C \([0-9]*\)/\1/p' ../include/nxsr_weights.h)
nxsr_host.o: ../../shaders/nxsr.cl ../../shaders/nx_common.h ../include/nxsr_weights.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -DSR_C=$(SR_C) -I../../shaders -O2 -c $< -o $@

osd_host.o: ../../shaders/osd.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

HOSTK = kernels_host.o fsr1_host.o smaa_host.o nxsr_host.o osd_host.o

gpu_emu: gpu_emu.c $(HOSTK) $(PORTABLE) $(HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-parameter -I../source -o $@ gpu_emu.c $(HOSTK) $(PORTABLE) -lm

ENGINE = ../source/nx_engine.c ../source/config.c $(PORTABLE)

console_sim: console_sim.c $(ENGINE) $(HEADERS) ../source/nx_engine.h
	gcc -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wno-unused-parameter \
	    -I../source -o $@ console_sim.c $(ENGINE) -lm

sim: console_sim
	./console_sim

run: host_test
	./host_test

emu: gpu_emu

clean:
	rm -f host_test gpu_emu console_sim *.o
