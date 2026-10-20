# Tests de la parte portable del plugin en el PC.
#   make run        -> host_test: ABI de kernargs + PM4 bien formado
#   make emu        -> gpu_emu: ejecuta el DCB con los kernels .cl compilados para x86
#   python3 plugin/test/test_gpu_path.py   -> compara gpu_emu con el modelo de referencia
CC = clang
CFLAGS ?= -O2 -Wall -Wextra
PORTABLE = ../source/pm4.c ../source/gcn_desc.c ../source/nx_pipeline.c
HEADERS = ../include/nx_shaders.h ../include/nxflow_weights.h

all: host_test gpu_emu

host_test: host_test.c $(PORTABLE) $(HEADERS)
	$(CC) $(CFLAGS) -I../source -o $@ host_test.c $(PORTABLE) -lm

kernels_host.o: ../../shaders/nxflow.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

fsr1_host.o: ../../shaders/fsr1.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

smaa_host.o: ../../shaders/smaa.cl ../../shaders/nx_common.h
	$(CC) -x cl -cl-std=CL1.2 -target x86_64-unknown-linux-gnu -DNX_HOST -I../../shaders -O2 -c $< -o $@

gpu_emu: gpu_emu.c kernels_host.o fsr1_host.o smaa_host.o $(PORTABLE) $(HEADERS)
	$(CC) $(CFLAGS) -Wno-unused-parameter -I../source -o $@ gpu_emu.c kernels_host.o fsr1_host.o smaa_host.o $(PORTABLE) -lm

run: host_test
	./host_test

emu: gpu_emu

clean:
	rm -f host_test gpu_emu *.o
