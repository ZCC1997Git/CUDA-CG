#ifndef _VECTOROPERATOR_
#define _VECTOROPERATOR_
#include <utility>

template <class Type, class OP, class... VEC>
__global__ void VectorOperation(OP op, unsigned L, Type* result, VEC... vec) {
    auto tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    for (auto i = tid; i < L; i += gridDim.x * blockDim.x)
        result[i] = op(i, vec...);
}

template <class Type>
__global__ void VectorDotVector(Type* sum_all,
                                Type* a,
                                Type* b,
                                unsigned int L) {
    size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    __shared__ Type block_sum[32];

    Type sum = 0;
    for (int i = tid; i < L; i += gridDim.x * blockDim.x)
        sum += a[i] * b[i];

    /*warp sum*/
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    auto warpID = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) {
        block_sum[warpID] = sum;
    }
    __syncthreads();

    if (warpID == 0) {
        sum = block_sum[threadIdx.x];
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        if (threadIdx.x == 0) atomicAdd(sum_all, sum);
    }
}

template <class Type>
__global__ void VectorNorm(Type* sum_all, Type* a, unsigned int L) {
    size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    __shared__ Type block_sum[32];

    Type sum = 0;
    for (int i = tid; i < L; i += gridDim.x * blockDim.x)
        sum += a[i] * a[i];

    /*warp sum*/
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    auto warpID = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) {
        block_sum[warpID] = sum;
    }
    __syncthreads();

    if (warpID == 0) {
        sum = block_sum[threadIdx.x];
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        if (threadIdx.x == 0) atomicAdd(sum_all, sum);
    }
}

#endif