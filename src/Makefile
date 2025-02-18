# Compiler
CC = gcc

# Target executable
TARGET = victordb

SRCS = main.c victor.c math.c
OBJS = $(SRCS:.c=.o)

CFLAGS = -std=c11 -Wall -Wextra -O3 -march=armv8-a+simd

LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

check-neon:
	@echo "Checking for NEON support..."
	@$(CC) -dM -E - < /dev/null | grep -q "__ARM_NEON" && echo "NEON supported" || echo "NEON not supported"

.PHONY: all clean check-neon
