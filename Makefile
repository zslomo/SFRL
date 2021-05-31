DEBUG=0

VPATH=./sfrl/
EXEC=sfrl_obj
OBJDIR=./obj/

CC=gcc
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors 

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ=softmax_layer.o iris.o blas.o activation.o maxtrix.o optimizer.o network.o base_layer.o batchnorm_layer.o dense_layer.o loss_layer.o metric.o data.o loader.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./sfrl/*/*.h) Makefile

all: obj backup results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

