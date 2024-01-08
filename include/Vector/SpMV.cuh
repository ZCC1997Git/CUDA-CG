#ifndef _SPMV_
#define _SPMV_

inline __constant__ static type_t* Addr[5];

template <class T>
__global__ void SpMV(T* V, T* result, unsigned int M, unsigned int Xnode) {
    /*just use one dim thread*/
    auto tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    auto step = gridDim.x * blockDim.x;

    /*used warp shuffle to reduce the access of V*/
    register T V_i, V_left, V_right;

    for (auto i = tid; i < M; i += step) {
        V_i = V[i + Xnode];
        V_left = __shfl_up_sync(0xFFFFFFFF, V_i, 1);
        V_right = __shfl_down_sync(0xFFFFFFFF, V_i, 1);
        if (threadIdx.x % 32 == 0) V_left = V[i - 1 + Xnode];
        if (threadIdx.x % 32 == 31) V_right = V[i + 1 + Xnode];

        result[i] = Addr[0][i] * V[i] + Addr[1][i] * V_left + Addr[2][i] * V_i +
                    Addr[3][i] * V_right + Addr[4][i] * V[i + Xnode + Xnode];
    }
}
#endif