#!/bin/sh

set -xe

CFLAGS="-O3 -Wall -Wextra -I./thirdparty/ `pkg-config --cflags raylib`"
LIBS="-lm `pkg-config --libs raylib` -lglfw -ldl -lpthread"

clang $CFLAGS -o adder adder.c $LIBS
clang $CFLAGS -o xor xor.c $LIBS
clang $CFLAGS -o img2nn img2nn.c $LIBS