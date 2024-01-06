#ifndef _SPMV_
#define _SPMV_

inline __constant__ static type_t* Addr[5];

template <class T>
__global__ void SpMV(T* V, T* result, unsigned int M, unsigned int Xnode) {
    /*just use one dim thread*/
    auto tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (auto i = tid; i < M; i += step) {
        result[i] = Addr[0][i] * V[i] + Addr[1][i] * V[i - 1 + Xnode] +
                    Addr[2][i] * V[i + Xnode] + Addr[3][i] * V[i + 1 + Xnode] +
                    Addr[4][i] * V[i + Xnode + Xnode];
    }
}
#endif