# AMBER - Ancient Metagenomic BinnER
# Makefile
#
# Build with: conda activate assembly && make

CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -mtune=native -Wall -Wextra -fopenmp

# Include paths
CXXFLAGS += -Iinclude -Isrc

# Eigen3 detection (for Cholesky decomposition)
ifdef CONDA_PREFIX
EIGEN_CFLAGS ?= -I$(CONDA_PREFIX)/include/eigen3
else
EIGEN_CFLAGS ?= $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
endif
CXXFLAGS += $(EIGEN_CFLAGS)

# HTSlib detection
# 1. Check CONDA_PREFIX (activated conda env)
# 2. Fall back to pkg-config
# 3. Allow override via HTSLIB_CFLAGS and HTSLIB_LIBS
ifdef CONDA_PREFIX
HTSLIB_CFLAGS ?= -I$(CONDA_PREFIX)/include
HTSLIB_LIBS ?= -L$(CONDA_PREFIX)/lib -lhts -Wl,-rpath,$(CONDA_PREFIX)/lib
else
HTSLIB_CFLAGS ?= $(shell pkg-config --cflags htslib 2>/dev/null || pkg-config --cflags hts 2>/dev/null)
HTSLIB_LIBS ?= $(shell pkg-config --libs htslib 2>/dev/null || pkg-config --libs hts 2>/dev/null || echo "-lhts")
endif
CXXFLAGS += $(HTSLIB_CFLAGS)

LDFLAGS := $(HTSLIB_LIBS) -lz -lpthread -lm

# libleidenalg for exact Leiden clustering (COMEBin matching)
# DISABLED: conda libleidenalg has different API than our patched version
# ifdef CONDA_PREFIX
# ifneq ($(wildcard $(CONDA_PREFIX)/lib/liblibleidenalg.so),)
# CXXFLAGS += -DUSE_LIBLEIDENALG -I$(CONDA_PREFIX)/include -I$(CONDA_PREFIX)/include/libleidenalg
# LDFLAGS += -L$(CONDA_PREFIX)/lib -ligraph -llibleidenalg -Wl,-rpath,$(CONDA_PREFIX)/lib
# $(info Using libleidenalg for exact Leiden clustering)
# endif
# endif

# Directories
SRCDIR := src
ALGDIR := src/algorithms
BINDIR := src/binner
CLIDIR := src/cli

# Target
TARGET := amber

# Debug mode
ifdef DEBUG
CXXFLAGS += -O0 -g -fno-omit-frame-pointer
endif

# Address sanitizer
ifdef ASAN
CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address
endif

# Source files
MAIN_SRCS := $(SRCDIR)/main.cpp

CLI_SRCS := \
	$(CLIDIR)/cli_common.cpp \
	$(CLIDIR)/cmd_bin.cpp \
	$(CLIDIR)/cmd_bin2.cpp \
	$(CLIDIR)/cmd_comebin.cpp \
	$(CLIDIR)/cmd_comebin_validate.cpp \
	$(CLIDIR)/cmd_leiden_debug.cpp \
	$(CLIDIR)/cmd_deconvolve.cpp \
	$(CLIDIR)/cmd_embed.cpp \
	$(CLIDIR)/cmd_polish.cpp \
	$(CLIDIR)/cmd_chimera.cpp \
	$(CLIDIR)/cmd_seeds.cpp

MODULE_SRCS := \
	$(SRCDIR)/embed.cpp \
	$(SRCDIR)/polish.cpp \
	$(SRCDIR)/chimera.cpp \
	$(SRCDIR)/bin2.cpp

ALG_SRCS := \
	$(ALGDIR)/fractal_features.cpp \
	$(ALGDIR)/kmer_features.cpp \
	$(ALGDIR)/coverage_features.cpp \
	$(ALGDIR)/damage_features.cpp \
	$(ALGDIR)/damage_profile.cpp \
	$(ALGDIR)/sequence_utils.cpp \
	$(ALGDIR)/pca.cpp \
	$(ALGDIR)/chimera_detector.cpp \
	$(ALGDIR)/ref_reader.cpp \
	$(ALGDIR)/parallel_coverage.cpp

BINNER_SRCS := \
	$(BINDIR)/binner.cpp

BIN2_CLUSTER_SRCS := \
	src/bin2/clustering/hnsw_knn_index.cpp \
	src/bin2/clustering/leiden_backend.cpp \
	src/bin2/clustering/edge_weighter.cpp

SEEDS_SRCS := \
	src/seeds/seed_generator.cpp

COMEBIN_SRCS := \
	src/comebin/comebin_binner.cpp

ALL_SRCS := $(MAIN_SRCS) $(CLI_SRCS) $(MODULE_SRCS) $(ALG_SRCS) $(BINNER_SRCS) $(BIN2_CLUSTER_SRCS) $(SEEDS_SRCS) $(COMEBIN_SRCS)

# Object files
OBJS := $(ALL_SRCS:.cpp=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo "Run './amber --help' to see usage"

# Compile rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Dependencies (auto-generated would be better, but this works for now)
$(SRCDIR)/main.o: $(SRCDIR)/main.cpp
$(CLIDIR)/cli_common.o: $(CLIDIR)/cli_common.cpp $(CLIDIR)/cli_common.h
$(CLIDIR)/cmd_bin.o: $(CLIDIR)/cmd_bin.cpp $(CLIDIR)/cli_common.h
$(CLIDIR)/cmd_deconvolve.o: $(CLIDIR)/cmd_deconvolve.cpp $(CLIDIR)/cli_common.h
$(CLIDIR)/cmd_embed.o: $(CLIDIR)/cmd_embed.cpp $(CLIDIR)/cli_common.h
$(CLIDIR)/cmd_polish.o: $(CLIDIR)/cmd_polish.cpp $(CLIDIR)/cli_common.h
$(CLIDIR)/cmd_chimera.o: $(CLIDIR)/cmd_chimera.cpp $(CLIDIR)/cli_common.h
$(SRCDIR)/embed.o: $(SRCDIR)/embed.cpp
$(SRCDIR)/polish.o: $(SRCDIR)/polish.cpp
$(SRCDIR)/chimera.o: $(SRCDIR)/chimera.cpp
$(BINDIR)/binner.o: $(BINDIR)/binner.cpp

clean:
	rm -f $(TARGET) $(OBJS)
	@echo "Cleaned build artifacts"

# Install target
PREFIX ?= /usr/local
install: $(TARGET)
	install -d $(PREFIX)/bin
	install -m 755 $(TARGET) $(PREFIX)/bin/

.PHONY: all clean install
