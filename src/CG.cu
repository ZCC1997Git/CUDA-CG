
#include <cuda.h>
#include <omp.h>
#include <ArrayAllocate/ArrayAllocate.cuh>
#include <Datatype.hpp>
#include <Vector/SpMV.cuh>
#include <Vector/VectorOperator.cuh>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

int SM;
void GetHardareInfo();

void CG(type_t** A,
        type_t* x,
        type_t* b,
        unsigned int Xnode,
        unsigned int Ynode) {
    std::cout << "Begin to calculate Ax=b using CG method" << std::endl;
    /*get GPU info and set block and grid*/
    GetHardareInfo();
    unsigned int LEN = Xnode * Ynode;
    dim3 grid = {1, 1, 1}, block = {1, 1, 1};
    grid.x = SM * 32;
    block.x = 1024;

    /*declar paramters*/
    type_t** gpu_A = Allocate2D<GPU, type_t>(5, LEN, A);
    type_t* gpu_x_all = Allocate1D<GPU, type_t>(LEN + 2 * Xnode, x - Xnode);
    type_t* gpu_b = Allocate1D<GPU, type_t>(LEN, b);
    type_t* gpu_r = Allocate1D<GPU, type_t>(LEN);
    type_t* gpu_r_tmp = Allocate1D<GPU, type_t>(LEN);
    type_t* gpu_tmp = Allocate1D<GPU, type_t>(LEN);
    type_t* gpu_p_all = Allocate1D<GPU, type_t>(LEN + 2 * Xnode);
    type_t* gpu_tmp_v = Allocate1D<GPU, type_t>(5);
    type_t* gpu_x = gpu_x_all + Xnode;
    type_t* gpu_p = gpu_p_all + Xnode;

    /*store the addr of A into the constant memory */
    cudaMemcpyToSymbol(Addr, gpu_A, 5 * sizeof(type_t*));

    /*set  x=0,r=0*/
    VectorOperation<<<1, 32>>>(
        [] __device__(int i, auto a) {
            a[i] = 0.f;
            return 0.f;
        },
        LEN, gpu_r, gpu_x);

    /*initializition*/
    /*tmp=A*x0*/
    SpMV<type_t><<<grid, block>>>(gpu_x_all, gpu_tmp, LEN, Xnode);

    /*r0=b-tmp*/
    VectorOperation<<<grid, block>>>(
        [] __device__(int i, auto a, auto b) { return a[i] - b[i]; }, LEN,
        gpu_r, gpu_b, gpu_tmp);

    /*p0=r0*/
    VectorOperation<<<grid, block>>>(
        [] __device__(int i, auto a) { return a[i]; }, LEN, gpu_p, gpu_r);

    /*computation*/
    auto t_begin = omp_get_wtime();
    int iteration = 0;
    while (true) {
        iteration++;

        /*set  gpu_tmp_v=0*/
        VectorOperation<<<1, 32>>>([] __device__(int i) { return 0.0f; }, 5,
                                   gpu_tmp_v);

        /*tmp=A*p*/
        SpMV<<<grid, block>>>(gpu_p_all, gpu_tmp, LEN, Xnode);

        /*auto tmp_v = VectorNorm(r, LEN);*/
        VectorNorm<<<grid, block>>>(gpu_tmp_v, gpu_r, LEN);

        /*tmp=p*p^T*/
        VectorDotVector<<<grid, block>>>(gpu_tmp_v + 1, gpu_p, gpu_tmp, LEN);

        /* x_k+1=x_k+alpha*p */
        /* r_tmp=r-alpha*tmp */
        VectorOperation<<<grid, block>>>(
            [gpu_tmp_v] __device__(int i, auto a, auto b, auto c, auto d,
                                   auto e) {
                auto alpha = gpu_tmp_v[0] / gpu_tmp_v[1];
                c[i] = d[i] - alpha * e[i];
                return a[i] + alpha * b[i];
            },
            LEN, gpu_x, gpu_x, gpu_p, gpu_r_tmp, gpu_r, gpu_tmp);

        /*auto beta = VectorNorm(r_tmp, LEN) / VectorNorm(r, LEN)*/
        VectorNorm<<<grid, block>>>(gpu_tmp_v + 2, gpu_r_tmp, LEN);

        /* p=r_tmp+beta*p */
        /* r=r_tmp */
        VectorOperation<<<grid, block>>>(
            [gpu_tmp_v] __device__(int i, auto a, auto b, auto c) {
                auto beta = gpu_tmp_v[2] / gpu_tmp_v[0];
                c[i] = a[i];
                return a[i] + beta * b[i];
            },
            LEN, gpu_p, gpu_r_tmp, gpu_p, gpu_r);

        /*the loop exit condition*/
        if (iteration % 10 == 0) {
            type_t tmp_v5 = 0;
            cudaMemcpy(&tmp_v5, gpu_tmp_v + 2, sizeof(type_t),
                       cudaMemcpyDeviceToHost);
            std::cout << "Iterator " << iteration << ": \t the absolute err is "
                      << std::sqrt(tmp_v5) << std::endl;

            if (std::sqrt(tmp_v5) < 1.e-6) {
                std::cout
                    << "The CG process has been Convergent, and CG  method "
                       "is completed !"
                    << std::endl;
                break;
            }
        }
    }

    auto t_end = omp_get_wtime();
    std::cout << "CG Runing time is " << t_end - t_begin << "s" << std::endl;

    cudaMemcpy(x, gpu_x, LEN * sizeof(type_t), cudaMemcpyDeviceToHost);
    Delete2D<GPU>(gpu_A);
    GpuDelete1D(gpu_x_all, gpu_b, gpu_r, gpu_r_tmp, gpu_tmp, gpu_p_all,
                gpu_tmp_v);
}

void GetHardareInfo() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::string cpuModel;

    while (std::getline(cpuinfo, line)) {
        if (line.substr(0, 10) == "model name") {
            cpuModel = line.substr(line.find(":") + 2);
            break;
        }
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    SM = deviceProp.multiProcessorCount;
    std::cout << "|-----------------------------------------------------|"
              << std::endl;
    std::cout << "|  GPU Model: " << deviceProp.name << "      |" << std::endl;
    std::cout << "|-----------------------------------------------------|"
              << std::endl;
    std::cout << "|  CPU Model: " << cpuModel << " |" << std::endl;
    std::cout << "|-----------------------------------------------------|"
              << std::endl;
    std::cout << std::endl;

    std::cout << "L2 Cache " << deviceProp.l2CacheSize / 1024 / 1024 << "M"
              << std::endl;
    std::cout << "maxBlocksPerMultiProcessor "
              << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    /*cuda core 3072,sm grid,cudacore_per_sm=block*/
    std::cout << "multiProcessorCount " << deviceProp.multiProcessorCount
              << std::endl;
    std::cout << "sharedMemPerMultiprocessor "
              << deviceProp.sharedMemPerMultiprocessor / 1024 << "K"
              << std::endl;
}