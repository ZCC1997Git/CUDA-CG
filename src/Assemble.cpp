#include <Setting.hpp>
#include <iostream>

void assemble() {
    /*set b*/
    for (int i = 0; i < Xnode; i++) {
        for (int j = 0; j < Ynode; j++) {
            int index = i * Ynode + j;
            if (i == 0 || i == Xnode - 1 || j == 0 || j == Ynode - 1)
                b[index] = 0;
            else b[index] = -h * h * Q;
        }
    }

    /*Set A*/
    for (int i = 0; i < Xnode; i++) {
        for (int j = 0; j < Ynode; j++) {
            int index = i * Ynode + j;
            if (i == 0 || i == Xnode - 1 || j == 0 || j == Ynode - 1)
                A[2][index] = 1;
            else {
                A[0][index] = 1;
                A[1][index] = 1;
                A[2][index] = -4;
                A[3][index] = 1;
                A[4][index] = 1;
            }
        }
    }
    std::cout << "Matrix A and vector b have been assembeled" << std::endl;
}
