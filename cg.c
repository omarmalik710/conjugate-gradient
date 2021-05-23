#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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

    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct stencil;
    stencil.size = 3;
    stencil.extent = stencil.size/2;
    stencil.stencil = (int*) malloc(stencil.size*stencil.size*sizeof(int));
    memcpy(stencil.stencil, my_stencil, sizeof(my_stencil));

    double error = 999;
    double q0 = dot(N,g,g);
    double tau, q1, beta;
    //printf("[INFO] q0 = %lf\n", q0);
    //while (error >= TOL) {

        apply_stencil(n, stencil, d, q);
        //tau = q0/dot(N,d,q);
        //u = vect_add(N, u, scal_mult(N,d,tau));
        //g = vect_add(N, g, scal_mult(N,q,tau));
        //q1 = dot(N,g,g);
        //beta = q1/q0;
        //d = vect_sub(N, scal_mult(N,d,beta), g);
        //q0 = q1;
    //}

    print_2dmesh(n, q);

    return 0;
}