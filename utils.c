#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "utils.h"

double* init_locald(int n, int rank, int numprocs, int chunk) {
    double h = (double) 1 / n;

    // Set different start and end points depending on rank. Ghost
    // values from neighbors will fill the rest (in another function).
    double* d;
    double istart, iend;
    if ( rank==0 ) {
        d = (double*) calloc((chunk+1)*(n+1),sizeof(double));
        istart = 0;
        iend = chunk;
    }
    else if ( rank==(numprocs-1) ) {
        d = (double*) calloc((chunk+1)*(n+1),sizeof(double));
        istart = 1;
        iend = chunk+1;
    }
    else {
        d = (double*) calloc((chunk+2)*(n+1),sizeof(double));
        istart = 1;
        iend = chunk+1;
    }

    // Initialize local d. No need to optimize this, since it's
    // only done once (at the start of the program).
    int i,j;
    double xi, yj;
    for (i=istart; i<iend; ++i) {
        xi = ((i-istart)+rank*chunk)*h;
        for (j=0; j<(n+1); ++j) {
            yj = j*h;
            // Account for boundary conditions.
            //if ((rank==0 && i==0) || (rank==(numprocs-1) && i==(chunk-1)) || j==0 || j==n) {
            if ((rank==0 && i==istart) || (rank==(numprocs-1) && i==chunk) || j==0 || j==n) {
                d[i*(n+1)+j] = 0.0;
            }
            else {
                d[i*(n+1)+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }

        }
    }

    return d;
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

    // Set different start and end points depending on rank. Ghost
    // values from neighbors will fill the rest (in another function).
    int istart;
    if ( rank==0 ) { istart = 0; }
    else { istart = 1; }

    // Initialize g. No need to optimize this, since it's
    // only done once (at the start of the program).
    for (int i=0; i<chunk; ++i) {
        for (int j=0; j<(n+1); ++j) {
            g[i*(n+1)+j] = -d[(i+istart)*(n+1)+j];
        }
    }
    return g;
}

void print_local2dmesh(int n, double* mesh, int rank, int numprocs, int chunk) {
    int istart = 0;
    int iend;
    if ( rank==0 || rank==(numprocs-1) ) { iend = chunk+1; }
    else { iend = chunk+2; }

    for (int i=istart; i<iend; ++i) {
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

void dot(int n, double* localv, double* localw,
        int rank, int chunk, MPI_Comm comm,
        double* result) {
    double localsum = 0.0;
    if (rank==0) {
        for (int i=0; i<chunk*(n+1); ++i) { localsum += localv[i]*localw[i]; }
    }
    else {
        for (int i=(n+1); i<(chunk+1)*(n+1); ++i) { localsum += localv[i]*localw[i]; }
    }
    //for (int i=(n+1); i<chunk*(n+1); ++i) { localsum += localv[i]*localw[i]; }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}
