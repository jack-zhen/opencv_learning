CC := g++
LD := g++
RM := rm -rfv


INCLUDE_DIR=-I/home/zhenaigong/program/opencv_4.2.0/include/opencv4 -Iinclude
LIB_DIR=-L/home/zhenaigong/program/opencv_4.2.0/lib -Lbin

SO_LDFLAGS := -shared -fPIC
CFLAGS :=  -Wall -fPIC -std=c++11 -g $(INCLUDE_DIR)

LDLIBS := $(LIB_DIR) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui 

DEMO := demo
DEMO_SRC := demo.cpp \
			src/quickdemo.cpp
DEMO_OBJ := $(DEMO_SRC:%.cpp=%.o)


$(DEMO): $(DEMO_OBJ)
	@echo "building" $@
	$(CC) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	@echo "complie " $@
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	@echo "Cleaning "
	@$(RM) $(DEMO) $(DEMO_OBJ)

.PHONY: clean
