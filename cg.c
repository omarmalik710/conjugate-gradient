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

    short stencil_width = 5;
    short extent = stencil_width/2;
    short stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    short stencil[5] = {-1, -1, 4, -1, -1};

    //double error = 999;
    double error = 0;
    while (error >= TOL) {
        for (int i=extent; i<(n+1)*(n+1)-extent; i++) {
            double result = 0;
            for (int j=0; j<stencil_width; j++) {
            for (int j=1; j<stencil_width; j++) {
                int index = i - extent + j*(n+1);
                result += stencil[j] * d[index];
            }
            q[i] = result;
        }
    }

    return 0;
}