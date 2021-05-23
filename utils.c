#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "utils.h"

//double* init_locald(int n, int rank, int numprocs, int chunk) {
d_struct* init_locald(int n, int rank, int numprocs, int chunk) {

    d_struct* locald = (d_struct*) malloc(sizeof(d_struct));
    locald->locald = (double*) malloc(chunk*(n+1)*sizeof(double));
    // Set different start and end points depending on rank. Ghost
    // values from neighbors will fill the rest (in another function).
    double istart, iend;
    if ( rank==0 ) {
        locald->top_pad = NULL;
        locald->bottom_pad = (double*) calloc(n+1,sizeof(double));
    }
    else if ( rank==(numprocs-1) ) {
        locald->top_pad = (double*) calloc(n+1,sizeof(double));
        locald->bottom_pad = NULL;
    }
    else {
        locald->top_pad = (double*) calloc(n+1,sizeof(double));
        locald->bottom_pad = (double*) calloc(n+1,sizeof(double));
    }

    // Initialize local d. No need to optimize this, since it's
    // only done once (at the start of the program).
    int i,j;
    double xi, yj;
    double h = (double) 1 / n;
    for (i=0; i<chunk; ++i) {
        xi = (i+rank*chunk)*h;
        for (j=0; j<(n+1); ++j) {
            yj = j*h;
            // Account for boundary conditions.
            //if ((rank==0 && i==0) || (rank==(numprocs-1) && i==(chunk-1)) || j==0 || j==n) {
            if ((rank==0 && i==0) || (rank==(numprocs-1) && i==(chunk-1)) || j==0 || j==n) {
                locald->locald[i*(n+1)+j] = 0.0;
            }
            else {
                locald->locald[i*(n+1)+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }

        }
    }

    return locald;
}

void exchange_boundaries(int n, double* d, int rank, int numprocs, int chunk) {
    MPI_Datatype rowtype;
    MPI_Type_contiguous(n+1, MPI_DOUBLE, &rowtype);
    MPI_Type_commit(&rowtype);

    MPI_Status status;
    MPI_Request request;
    int tags[numprocs-1];
    for (int i=0; i<(numprocs-1); ++i) { tags[i] = i; }

    double *sendbuf, *recvbuf;
    if ( rank==0 ) {
        sendbuf = d+(chunk-1)*(n+1);
        recvbuf = d+chunk*(n+1);
        //MPI_Isend(sendbuf, 1, rowtype, rank+1, tags[rank], MPI_COMM_WORLD, &status);
        //MPI_Recv(recvbuf, 1, rowtype, rank+1, tags[rank+1], MPI_COMM_WORLD, &status);
        MPI_Sendrecv(sendbuf, 1, rowtype, rank+1, tags[rank],
                    recvbuf, 1, rowtype, rank+1, tags[rank+1],
                    MPI_COMM_WORLD, &status);
    }
    else if ( rank==(numprocs-1) ) {
        sendbuf = d+(n+1);
        recvbuf = d;
        //MPI_Isend(d+(n+1), 1, rowtype, rank-1, tags[rank], MPI_COMM_WORLD, &status);
        //MPI_Recv(d, 1, rowtype, rank-1, tags[rank-1], MPI_COMM_WORLD, &status);
        MPI_Sendrecv(sendbuf, 1, rowtype, rank-1, tags[rank],
                    recvbuf, 1, rowtype, rank-1, tags[rank-1],
                    MPI_COMM_WORLD, &status);
    }
    else {
        sendbuf = d+(n+1);
        recvbuf = d;
        //MPI_Isend(d+(n+1), 1, rowtype, rank-1, tags[rank], MPI_COMM_WORLD, &status);
        //MPI_Recv(d, 1, rowtype, rank-1, tags[rank-1], MPI_COMM_WORLD, &status);
        MPI_Sendrecv(sendbuf, 1, rowtype, rank-1, tags[rank],
                    recvbuf, 1, rowtype, rank-1, tags[rank-1],
                    MPI_COMM_WORLD, &status);

        sendbuf = d+(chunk-1)*(n+1);
        recvbuf = d+chunk*(n+1);
        //MPI_Isend(d+(i-1)*(n+1), 1, rowtype, rank+1, tags[rank], MPI_COMM_WORLD, &status);
        //MPI_Recv(d+i*(n+1), 1, rowtype, rank+1, tags[rank+1], MPI_COMM_WORLD, &status);
        MPI_Sendrecv(sendbuf, 1, rowtype, rank+1, tags[rank],
                    recvbuf, 1, rowtype, rank+1, tags[rank+1],
                    MPI_COMM_WORLD, &status);
    }
}

double* init_localg(int n, double* d, int rank, int chunk) {
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

void print_local2dmesh(int rows, int cols, double* mesh, int rank) {
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            printf("([%d] %lf) ", rank, mesh[i*cols+j]);
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

void dot(int rows, int cols, double* localv, double* localw,
        int rank, MPI_Comm comm, double* result) {
    double localsum = 0.0;
    for (int i=0; i<rows*cols; ++i) { localsum += localv[i]*localw[i]; }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}
