CC=g++ -O3 -fopenmp -std=c++17 -mavx2
SRCS=$(wildcard *.cpp */*.cpp)
OBJS=$(patsubst %.cpp, %.o, $(SRCS))

INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include -I/home/research/code/SOSD/competitors/stx-btree-0.9/include -I/home/research/code/SOSD/competitors/PGM-index/include
LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
NAME=$(wildcard *.cpp)
TARGET=$(patsubst %.cpp, %, $(NAME))

$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)

%.o:%.cpp
	$(CC) -o $@ -c $< -g $(INCLUDE)

clean:
	rm -rf $(TARGET) $(OBJS)