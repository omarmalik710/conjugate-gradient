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

    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct stencil;
    stencil.size = 3;
    stencil.extent = stencil.size/2;
    stencil.stencil = (int*) malloc(stencil.size*stencil.size*sizeof(int));
    memcpy(stencil.stencil, my_stencil, sizeof(my_stencil));

    double* u = (double*) calloc(N,sizeof(double));
    double* d = init_d(n);
    double* g = init_g(n, d);
    double* q = (double*) malloc(N*sizeof(double));
    double* tau_d = (double*) malloc(N*sizeof(double));
    //print_2dmesh(n, d);
    //print_2dmesh(n, g);

    int i;
    double error = 999;
    double tau, q1, beta;
    double q0 = dot(N,g,g);
    printf("[INFO] q0 = %lf\n", q0);
    for (int iter=0; iter<MAX_ITERS; iter++) {
        apply_stencil(n, stencil, d, q);
        tau = q0/dot(N,d,q);
        //printf("[INFO] tau = %lf\n", tau);
        for (i=0; i<N; ++i) { u[i] += tau*d[i]; }
        for (i=0; i<N; ++i) { g[i] += tau*q[i]; }
        q1 = dot(N,g,g);
        beta = q1/q0;
        for (i=0; i<N; ++i) { d[i] = beta*d[i] - g[i]; }
        q0 = q1;
    }

    //print_2dmesh(n, q);
    //putchar('\n');
    //print_2dmesh(n, u);
    //putchar('\n');
    //print_2dmesh(n, g);
    //putchar('\n');
    //print_2dmesh(n, d);

    double norm_g = sqrt(dot(N,g,g));
    printf("[INFO] norm_g = %.16lf\n", norm_g);

    return 0;
}