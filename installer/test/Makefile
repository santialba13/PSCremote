# Tests de la app en el PC (con AddressSanitizer y UndefinedBehaviorSanitizer):
#   make -C installer/test           -> test_core + test_app
#   make -C installer/test preview   -> capturas de todas las pantallas (ui_*.ppm)
CC = gcc
SAN = -g -fsanitize=address,undefined
CFLAGS = -Wall -Wno-unknown-pragmas -Wno-format-truncation -Wno-unused-function -Ifake -I../source
SRC = ../source/inst_core.c ../../plugin/source/config.c

all: test_core test_app
	./test_core
	ASAN_OPTIONS=detect_leaks=0 ./test_app

test_core: test_core.c $(SRC)
	$(CC) $(SAN) $(CFLAGS) -o $@ test_core.c $(SRC)

test_app: test_app.c ../source/main.c $(SRC)
	$(CC) $(SAN) $(CFLAGS) -DROOT='"./tfs"' -o $@ test_app.c $(SRC) -lm

preview: preview.c ../source/main.c $(SRC)
	$(CC) -O2 $(CFLAGS) -o $@ preview.c $(SRC) -lm
	./preview $(ICONS)

clean:
	rm -f test_core test_app preview ui_*.ppm

.PHONY: all clean
