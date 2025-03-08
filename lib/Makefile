CC = gcc

TARGET = testdb

LIBNAME = libvictor.so

SRCS = test.c index.c index_flat.c math.c mem.c method.c vector.c
OBJS = $(SRCS:.c=.o)

# Compiler flags (Optimized for ARMv8-A / AArch64)
CFLAGS = -g -std=c11 -Wall -Wextra -O3 -march=armv8-a+simd -fPIC

LDFLAGS = -lm -lpthread

all: $(TARGET) $(LIBNAME)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBNAME): index.o index_flat.o math.o mem.o method.o vector.o
	$(CC) -shared -o $(LIBNAME) index.o index_flat.o math.o mem.o method.o vector.o $(LDFLAGS)


clean:
	rm -f $(OBJS) $(TARGET) $(LIBNAME)

check-neon:
	@echo "Checking for NEON support..."
	@$(CC) -dM -E - < /dev/null | grep -q "__ARM_NEON" && echo "NEON supported" || echo "NEON not supported"

.PHONY: all clean check-neon


