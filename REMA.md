# 二维热扩散方程求解代码

* 语言C++，linux环境，需要g++或其他C++编译器，如果需要替换C++编译器，需要修改makefile中的compiler 参数
* 代码编译 make，生成可执行文件HeatDifu
* 代码运行 make run,或者直接./HeatDifu
* 结果保存再output文件中
* 结果绘图需要python环境，直接make plot，绘图结果保存再output文件夹里
* 清除编译文件make clean
* 清除编译文件和计算结果 make clean_all
* 更改网格数量需要更改Setting.hpp中Xnode的值

srun -N 1 -n 16 -p cpu --pty /bin/bash
module load gcc/12.2.0
export OMP_NUM_THREADS=8
make run
