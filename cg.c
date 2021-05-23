#include <stdio.h>
#include <stdlib.h>
//#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv) {

    int n = atoi(argv[1]);
    //int N = (n-1)*(n-1);
    int N = (n+1)*(n+1);

    double* u = (double*) calloc((n+1)*(n+1), sizeof(double));
    double* q = (double*) calloc((n+1)*(n+1), sizeof(double));

    double* d = init_d(n);
    double* g = init_g(n, d);
    print_2dmesh(n, d);
    print_2dmesh(n, g);

    double q0 = dot(N,g,g);

    short stencil[5] = {-1, -1, 4, -1, -1};
    short stencil_width = 5;
    short extent = stencil_width/(2*2);

    //double error = 999;
    double error = 0;
    double result;
    int index;
    while (error >= TOL) {
        for (int i=extent; i<(n+1)-extent; ++i) {
            for (int j=extent; j<(n+1)-extent; ++j) {
                result = 0;
                for (int k=0; k<stencil_width; ++k) {
                    index = i*(n+1) + (j-extent+k);
                    result += stencil[k] * d[index];
                }
                q[i*(n+1)+j] = result;
            }
        }
    }

    return 0;
}