EXECUTABLE := coo_multiply

CU_FILES   := coo_multiply.cu

CU_DEPS    :=

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

OBJS=$(OBJDIR)/coo_multiply.o


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) *.o

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

