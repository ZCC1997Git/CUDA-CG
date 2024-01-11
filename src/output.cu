#include <ArrayAllocate/ArrayAllocate.cuh>
#include <Setting.hpp>
#include <fstream>
#include <iostream>
using namespace std;

void Output() {
    cout << "Outputing the results (in directory ./output)" << endl;
    /*output the result of solution*/
    int n = Xnode;
    type_t* node_x = Allocate1D<CPU, type_t>(n * n);
    type_t* node_y = Allocate1D<CPU, type_t>(n * n);

    for (int i = 0; i < n * n; i++) {
        node_x[i] = (R / (n - 1)) * (int)(i / n);
        node_y[i] = (R / (n - 1)) * (i % n);
    }

    ofstream outfile1("./output/xr.dat");
    ofstream outfile2("./output/yr.dat");
    ofstream outfile3("./output/solution.dat");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            outfile1 << node_x[i * n + j] << "\t";
            outfile2 << node_y[i * n + j] << "\t";
            outfile3 << x[i * n + j] << "\t";
        }
        outfile1 << endl;
        outfile2 << endl;
        outfile3 << endl;
    }

    outfile1.close();
    outfile2.close();
    outfile3.close();

    Delete1Ds<CPU>(node_x, node_y);
}