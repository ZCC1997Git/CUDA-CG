compiler  		=g++
nvcc			=nvcc

ifeq ($(mode),debug)
	cflags    	+=-std=c++17 -g3 -ggdb  -fopenmp
	linkflags 	+=-Xcompiler  -fopenmp
	nvflags		+=-gencode=arch=compute_80,code=compute_80 -G -O0 --extended-lambda -std=c++17
else
	cflags    	+=-std=c++17 -Ofast -march=native -fopt-info-vec -fopt-info-inline  -fopenmp
	linkflags 	+=-Xcompiler -fopenmp 
	nvflags		+=-gencode=arch=compute_80,code=compute_80 --extended-lambda -O3 -std=c++17
endif

SrcDir	  		=./src
ObjDir    		=./obj
HeadRoot		=./include
HeadDir         +=$(HeadRoot) $(foreach dir,$(HeadRoot),$(wildcard $(dir)/*))
source    		=$(foreach dir,$(SrcDir),$(wildcard $(dir)/*.cpp $(dir)/*.cu))
head      		=$(foreach dir,$(HeadDir),$(wildcard $(dir)/*.hpp $(dir)/*.cuh))
object    		=$(patsubst %.cpp,$(ObjDir)/%.o,$(patsubst %.cu,$(ObjDir)/%.cu.o,$(notdir $(source))))

target 	  		=HeatDifu
NO_COLOR		=\033[0m
OK_COLOR		=\033[32;01m


$(target):$(object) $(head)
	$(nvcc) -o $(target) $(object) $(linkflags) $(lib)
	@printf "$(OK_COLOR)Compiling Is Successful!\nExecutable File: $(target) $(NO_COLOR)\n"

$(ObjDir)/%.o:$(SrcDir)/%.cpp $(head)
	$(compiler) -c $(cflags) $< -o $@ -I $(HeadRoot)

$(ObjDir)/%.cu.o:$(SrcDir)/%.cu $(head)
	$(nvcc) -c $(nvflags) $< -o $@ -I $(HeadRoot)

.PHONY:run 
run:$(target)
	@printf "$(OK_COLOR)$(target) is executing $(NO_COLOR)\n"
	./$(target)

.PHONY:clean	 
clean:
	-rm $(object) $(target)

.PHONY:clean_all	 
clean_all:
	-rm $(object) $(target) ./output/*

.PHONY:plot
plot:
	python3 Plot.py