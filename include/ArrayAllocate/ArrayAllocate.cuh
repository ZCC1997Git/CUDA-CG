#ifndef __ARRARALLOCATE_
#define __ARRARALLOCATE_
#include <cuda_runtime.h>
#include <iostream>

/**
 * define the device
 */
enum device { CPU = 0, GPU = 1 };

template <device Device, class T>
T** Allocate2D(int M, int N, T** ref [[maybe_unused]] = nullptr) {
    if constexpr (Device == CPU) {
        T** tmp;
        tmp = new T*[M];
        tmp[0] = new T[M * N];
        for (int i = 0; i < M * N; i++)
            tmp[0][i] = 0;

        for (int i = 0; i < M; i++)
            tmp[i] = tmp[0] + i * N;

        return tmp;
    } else if constexpr (Device == GPU) {
        T** tmp = new T*[M];
        T** tmp_gpu = nullptr;
        cudaMalloc(&tmp_gpu, M * sizeof(T*));

        T* device_tmp = nullptr;
        cudaMalloc(&device_tmp, M * N * sizeof(T));

        if (ref != nullptr)
            cudaMemcpy(device_tmp, ref[0], M * N * sizeof(T),
                       cudaMemcpyHostToDevice);

        for (int i = 0; i < M; i++)
            tmp[i] = device_tmp + i * N;

        cudaMemcpy(tmp_gpu, tmp, M * sizeof(T*), cudaMemcpyHostToDevice);
        delete[] tmp;
        return tmp_gpu;
    } else {
        std::cerr << "unsupport device during alloc" << std::endl;
    }
}

template <device Device, class T>
void Delete2D(T** buf) {
    if constexpr (Device == CPU) {
        if (buf[0]) delete[] buf[0];
        if (buf) delete[] buf;
    } else if constexpr (Device == GPU) {
        T* addr = nullptr;
        cudaMemcpy(&addr, buf, sizeof(T*), cudaMemcpyDeviceToHost);
        cudaFree(addr);
        cudaFree(buf);
    } else {
        std::cerr << "unsupport device during free" << std::endl;
    }
}

template <device Device, class T>
T* Allocate1D(int M, T* ref [[maybe_unused]] = nullptr) {
    if constexpr (Device == CPU) {
        T* tmp = new T[M];
        /*initialize the value to be 0*/
        for (int i = 0; i < M; i++)
            tmp[i] = 0;
        return tmp;
    } else if constexpr (Device == GPU) {
        T* tmp;
        cudaMalloc(&tmp, M * sizeof(T));
        if (ref != nullptr)
            cudaMemcpy(tmp, ref, M * sizeof(T), cudaMemcpyHostToDevice);
        return tmp;
    } else {
        std::cerr << "unsupport device during malloc" << std::endl;
    }
}

template <device Device, class T>
void Delete1D(T* buf) {
    if constexpr (Device == CPU) delete[] buf;
    else if constexpr (Device == GPU) cudaFree(buf);
    else {
        std::cerr << "unsupport device during free" << std::endl;
    }
}

template <class... Vec>
void GpuDelete1D(Vec... vec) {
    (Delete1D<GPU>(vec), ...);
}
#endif