#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv) {

    int n = atoi(argv[1]);
    int N = (n+1)*(n+1);

    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    if ((n+1)%numprocs != 0) {
        printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
        exit(1);
    }

    int tags[numprocs-1];
    for (int i=0; i<(numprocs-1); ++i) { tags[i] = i; }

    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct stencil;
    stencil.size = 3;
    stencil.extent = stencil.size/2;
    stencil.stencil = (int*) malloc(stencil.size*stencil.size*sizeof(int));
    memcpy(stencil.stencil, my_stencil, sizeof(my_stencil));

    int chunk = (n+1) / numprocs;
    double* localu = (double*) calloc(chunk*(n+1),sizeof(double));
    double* localq = (double*) malloc(chunk*(n+1)*sizeof(double));
    d_struct* locald = init_locald(n, rank, numprocs, chunk);
    exchange_boundaries(n, locald, rank, numprocs, chunk);
    double* localg = init_localg(n, locald->locald, rank, chunk);
    //print_local2dmesh(n, localu, rank, chunk);
    //putchar('\n');
    //print_local2dmesh(1, n+1, locald->top_pad, rank);
    //putchar('\n');
    //print_local2dmesh(chunk, n+1, locald->locald, rank);
    //putchar('\n');
    //print_local2dmesh(1, n+1, locald->bottom_pad, rank);
    //putchar('\n');
    //print_local2dmesh(chunk, n+1, localg, rank);
    //double q0;
    //dot(chunk, n+1, localg, localg, rank, MPI_COMM_WORLD, &q0);
    //printf("[RANK %d] q0 = %lf\n", rank, q0);

    apply_stencil(n, stencil, locald, localq, rank, numprocs, chunk);
    print_local2dmesh(chunk, n+1, localq, rank);

    //int i;
    //double error = 999;
    //double tau, q1, beta;
    //double q0 = dot(N,g,g);
    //printf("[INFO] q0 = %lf\n", q0);
    //for (int iter=0; iter<MAX_ITERS; iter++) {
    //    apply_stencil(n, stencil, d, q);
    //    tau = q0/dot(N,d,q);
    //    //printf("[INFO] tau = %lf\n", tau);
    //    for (i=0; i<N; ++i) { u[i] += tau*d[i]; }
    //    for (i=0; i<N; ++i) { g[i] += tau*q[i]; }
    //    q1 = dot(N,g,g);
    //    beta = q1/q0;
    //    for (i=0; i<N; ++i) { d[i] = beta*d[i] - g[i]; }
    //    q0 = q1;
    //}

    //print_2dmesh(n, q);
    //putchar('\n');
    //print_2dmesh(n, u);
    //putchar('\n');
    //print_2dmesh(n, g);
    //putchar('\n');
    //print_2dmesh(n, d);

    //double norm_g = sqrt(dot(N,g,g));
    //printf("[INFO] norm_g = %.16lf\n", norm_g);

    MPI_Finalize();

    return 0;
}