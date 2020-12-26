PROJECT   =    segmentation
OBJ       :=   main.o  


CXX       ?=   g++
CC        ?=   gcc

# linking libraries of OpenCV
LDFLAGS   = $(shell pkg-config --libs opencv)

# linking libraries of DNNDK 
LDFLAGS   +=  -ln2cube -lpthread -lhineon

CUR_DIR   =   $(shell pwd)
SRC       =   $(CUR_DIR)/src
BUILD     =   $(CUR_DIR)/build
MODEL	  =   $(CUR_DIR)/model
VPATH     =   $(SRC)
MODEL = $(CUR_DIR)/model/dpu_segmentation_0.elf

ARCH    =   $(shell uname -m | sed -e s/arm.*/armv71/ -e s/aarch64.*/aarch64/)
CFLAGS :=   -O2 -Wall -Wpointer-arith -std=c++17 -ffast-math -I../common/
CFLAGS += -mcpu=cortex-a53 -I../common/
 
all: $(BUILD) $(PROJECT)
 
$(PROJECT): $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)
 
%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@ 

clean:
	$(RM) -r $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) : 
	-mkdir -p $@
