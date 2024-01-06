#include <ArrayAllocate/ArrayAllocate.cuh>
#include <Setting.hpp>

void finalization() {
    Delete2D<CPU>(A);
    Delete1D<CPU>(x_all);
    Delete1D<CPU>(b);
}