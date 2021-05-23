#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "utils.h"

double* init_locald(int n, int rank, int numprocs, int chunk) {
    double h = (double) 1 / n;
    double *d = (double*) malloc(chunk*(n+1)*sizeof(double));

    // Initialize d. No need to optimize this, since it's
    // only done once (at the start of the program).
    double xi, yj;
    for (int i=0; i<chunk; ++i) {
        xi = (i+rank*chunk)*h;
        for (int j=0; j<(n+1); ++j) {
            yj = j*h;
            // Account for boundary conditions.
            if ((rank==0 && i==0) || (rank==(numprocs-1) && i==(chunk-1)) || j==0 || j==n) {
                d[i*(n+1)+j] = 0.0;
            }
            else {
                d[i*(n+1)+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }

        }
    }
    return d;
}

double* init_localg(int n, double* d, int chunk) {
    double* g = (double*) malloc(chunk*(n+1)*sizeof(double));

    // Initialize g. No need to optimize this, since it's
    // only done once (at the start of the program).
    for (int i=0; i<chunk; ++i) {
        for (int j=0; j<(n+1); ++j) {
            g[i*(n+1)+j] = -d[i*(n+1)+j];
        }
    }
    return g;
}

void print_local2dmesh(int n, double* mesh, int rank, int chunk) {
    for (int i=0; i<chunk; ++i) {
        for (int j=0; j<(n+1); ++j) {
            printf("([%d] %lf) ", rank, mesh[i*(n+1)+j]);
        }
        putchar('\n');
    }
}

void apply_stencil(int n, stencil_struct my_stencil, double* src, double* dest) {
    int stencil_size = my_stencil.size;
    int extent = my_stencil.extent;
    int* stencil = my_stencil.stencil;

    double result;
    int i,j,l,m;
    int index;
    for (i=extent; i<(n+1)-extent; ++i) {
        for (j=extent; j<(n+1)-extent; ++j) {

            result = 0;
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * src[index];
                }
            }
            dest[i*(n+1)+j] = result;
        }
    }
}

void dot(int N, double* localv, double* localw, MPI_Comm comm, double* result) {
    double localsum = 0.0;
    for (int i=0; i<N; ++i) { localsum += localv[i]*localw[i]; }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}
