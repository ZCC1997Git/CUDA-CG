#include <ArrayAllocate/ArrayAllocate.cuh>
#include <Setting.hpp>

void initialization() {
    /*determine the size of matrix*/
    unsigned int SIZE = Xnode * Ynode;

    /*allcate matrix A*/
    A = Allocate2D<CPU, type_t>(5, SIZE);

    /*allocate vector X*/
    x_all = Allocate1D<CPU, type_t>(SIZE + 2 * Xnode);
    x = x_all + Xnode;

    /*allocate vector b*/
    b = Allocate1D<CPU, type_t>(SIZE);
}