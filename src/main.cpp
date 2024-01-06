
#include <Setting.hpp>
#include <iostream>

void initialization();
void finalization();
void assemble();
void CG(type_t** A,
        type_t* x,
        type_t* b,
        unsigned int Xnode,
        unsigned int Ynode);
void Output();

int main() {
    initialization();
    assemble();
    CG(A, x, b, Xnode, Ynode);
    Output();
    finalization();
    return 0;
}