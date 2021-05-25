CC = mpicc
CFLAGS = -std=c99 -g -O3 -march=native
LIBS = -lmpi -lm

BIN = cg

all: $(BIN)

cg: cg.c utils.c utils.h
	$(CC) $(CFLAGS) cg.c utils.c -o $(BIN) $(LIBS)

clean:
	$(RM) $(BIN)
