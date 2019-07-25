.PHONY: all clean test swig

CXX = g++
CXX_FLAGS = -O2 -fPIC 
IFLAGS = -I/usr/include/python3.4
MODULE = bfcheck

all: swig

%.o : %.cpp 
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(IFLAGS)

$(MODULE)_wrap.cpp: $(MODULE).i
	swig -c++ -python -o $(MODULE)_wrap.cpp $(MODULE).i

_$(MODULE).so: $(MODULE).o $(MODULE)_wrap.o
	$(CXX) -shared $^ -o _$(MODULE).so

clean:
	@rm -rf *.o main *~ core \
	$(MODULE).py $(MODULE)_wrap.cpp \
	_$(MODULE).so

swig: _$(MODULE).so

test: _$(MODULE).so
	@python3 main.py
