# Compiler and flags
CC = gcc
CFLAGS = -O2

# Target executable
TARGET = read_matrix

# Source files
SRCS = read_matrix.cpp mmio.c

# Object files (derived from source files)
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

