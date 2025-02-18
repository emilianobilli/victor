CC = gcc

TARGET = testdb

LIBNAME = libvictor.so

SRCS = test.c victor.c math.c
OBJS = $(SRCS:.c=.o)

# Compiler flags (Optimized for ARMv8-A / AArch64)
CFLAGS = -g -std=c11 -Wall -Wextra -O3 -march=armv8-a+simd -fPIC

LDFLAGS = -lm

all: $(TARGET) $(LIBNAME)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBNAME): victor.o math.o
	$(CC) -shared -o $(LIBNAME) victor.o math.o $(LDFLAGS)


clean:
	rm -f $(OBJS) $(TARGET) $(LIBNAME)

check-neon:
	@echo "Checking for NEON support..."
	@$(CC) -dM -E - < /dev/null | grep -q "__ARM_NEON" && echo "NEON supported" || echo "NEON not supported"

.PHONY: all clean check-neon

