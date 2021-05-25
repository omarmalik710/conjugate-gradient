#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "utils.h"

d_struct* init_locald(int n, int chunklength, int wrank, MPI_Comm cartcomm, int cartsize) {
    d_struct* locald = (d_struct*) malloc(sizeof(d_struct));
    locald->locald = (double*) malloc(chunklength*chunklength*sizeof(double));
    int coords[2];
    MPI_Cart_coords(cartcomm, wrank, 2, coords);
    int carti = coords[0];
    int cartj = coords[1];

    // Initialize top and bottom pads in cartesian topology. First and
    // last rows of procs are padded in one direction, while inner rows
    // are padded in two.
    if ( carti==0 ) {
        locald->top_pad = NULL;
        locald->bottom_pad = (double*) calloc(chunklength,sizeof(double));
    }
    else if ( carti==(cartsize-1) ) {
        locald->top_pad = (double*) calloc(chunklength,sizeof(double));
        locald->bottom_pad = NULL;
    }
    else {
        locald->top_pad = (double*) calloc(chunklength,sizeof(double));
        locald->bottom_pad = (double*) calloc(chunklength,sizeof(double));
    }

    // Initialize left and right pads in cartesian topology. First and
    // last columns of procs are padded in one direction, while inner
    // columns are padded in two.
    if ( cartj==0 ) {
        locald->left_pad = NULL;
        locald->right_pad = (double*) calloc(chunklength,sizeof(double));
    }
    else if ( cartj==(cartsize-1) ) {
        locald->left_pad = (double*) calloc(chunklength,sizeof(double));
        locald->right_pad = NULL;
    }
    else {
        locald->left_pad = (double*) calloc(chunklength,sizeof(double));
        locald->right_pad = (double*) calloc(chunklength,sizeof(double));
    }

    // Initialize local d. No need to optimize this, since it's
    // only done once (at the start of the program).
    int i,j;
    double xi, yj;
    double h = (double) 1 / n;
    for (i=0; i<chunklength; ++i) {
        xi = (i+carti*chunklength)*h;
        for (j=0; j<chunklength; ++j) {
            yj = (j+cartj*chunklength)*h;
            // Account for boundary conditions.
            if ( (carti==0 && i==0) || (carti==(cartsize-1) && i==(chunklength-1)) ||
                 (cartj==0 && j==0) || (cartj==(cartsize-1) && j==(chunklength-1)) ) {
                locald->locald[i*chunklength+j] = 0.0;
            }
            else {
                locald->locald[i*chunklength+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }
        }
    }

    return locald;
}

void exchange_boundaries(int n, d_struct* locald, int rank, int numprocs, int chunk, MPI_Request* requests) {
    MPI_Datatype rowtype;
    MPI_Type_contiguous(n+1, MPI_DOUBLE, &rowtype);
    MPI_Type_commit(&rowtype);

    int tags[numprocs];
    for (int i=0; i<numprocs; ++i) { tags[i] = i; }

    double *sendbuf, *recvbuf;
    // Send from / recv to bottom boundary. First
    // and intermediate procs will do this.
    if (locald->bottom_pad != NULL) {
        sendbuf = locald->locald + (chunk-1)*(n+1);
        recvbuf = locald->bottom_pad;
        MPI_Isend(sendbuf, 1, rowtype, rank+1, tags[rank], MPI_COMM_WORLD, requests+rank);
        MPI_Irecv(recvbuf, 1, rowtype, rank+1, tags[rank+1], MPI_COMM_WORLD, requests+(rank+1));
    }
    // Send from / recv to top boundary. Last
    // and intermediate procs will do this.
    if (locald->top_pad != NULL) {
        sendbuf = locald->locald;
        recvbuf = locald->top_pad;
        MPI_Isend(sendbuf, 1, rowtype, rank-1, tags[rank], MPI_COMM_WORLD, requests+rank);
        MPI_Irecv(recvbuf, 1, rowtype, rank-1, tags[rank-1], MPI_COMM_WORLD, requests+(rank-1));
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

void print_local2dmesh(int rows, int cols, double* mesh, int wrank, MPI_Comm cartcomm) {
    int coords[2];
    MPI_Cart_coords(cartcomm, wrank, 2, coords);
    int carti = coords[0];
    int cartj = coords[1];

    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            if (mesh != NULL) {
                printf("([%d (%d,%d)] %lf) ", wrank, carti, cartj, mesh[i*cols+j]);
            }
        }
        putchar('\n');
    }
}

void apply_stencil(int n, stencil_struct* my_stencil, d_struct* locald, double* localq, int rank, int numprocs, int chunk) {
    int stencil_size = my_stencil->size;
    int extent = my_stencil->extent;
    int* stencil = my_stencil->stencil;

    // Exchange top and/ or bottom padded boundaries
    // using nonblocking communication.
    MPI_Request requests[numprocs];
    //MPI_Status statuses[numprocs];
    MPI_Status status;
    exchange_boundaries(n, locald, rank, numprocs, chunk, requests);

    // Apply stencil on inner points while exchanging
    // the padded boundaries above.
    double result;
    int i,j,l,m;
    int index;
    for (i=extent; i<chunk-extent; ++i) {
        for (j=extent; j<(n+1)-extent; ++j) {

            result = 0;
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            localq[i*(n+1)+j] = result;
        }
    }

    // Wait for receive of padded boundaries from previous (rank-1)
    // and/or next (rank+1) procs to finish before applying the
    // stencil on the outer points.
    if (locald->top_pad != NULL) {
        MPI_Wait(requests+(rank-1), &status);
    }
    if (locald->bottom_pad != NULL) {
        MPI_Wait(requests+(rank+1), &status);
    }

    // Apply stencil on outer points. First proc uses bottom pad,
    // last proc uses top pad, and intermediate procs use both pads.
    //// First row (i==0).
    if (locald->top_pad != NULL) {
        for (j=extent; j<(n+1)-extent; ++j) {
            result = 0;
            // Handle left, bottom, and right neighbors as usual.
            for (l=1; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (-extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            // Use top_pad for the top neighbors (l==0).
            for (m=0; m<stencil_size; ++m) {
                index = j - extent + m;
                result += stencil[m] * locald->top_pad[index];
            }
            localq[j] = result;
        }
    }
    //// Last row (i==chunk-extent).
    if (locald->bottom_pad != NULL) {
        for (j=extent; j<(n+1)-extent; ++j) {
            result = 0;
            // Handle left, top, and right neighbors as usual.
            for (l=0; l<stencil_size-1; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            // Use bottom_pad for the bottom neighbors (l==stencil_size-1).
            for (m=0; m<stencil_size; ++m) {
                index = j - extent + m;
                result += stencil[l*stencil_size+m] * locald->bottom_pad[index];
            }
            localq[i*(n+1)+j] = result;
        }
    }
}

void dot(int rows, int cols, double* localv, double* localw,
        int rank, MPI_Comm comm, double* result) {
    double localsum = 0.0;
    for (int i=0; i<rows*cols; ++i) { localsum += localv[i]*localw[i]; }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}
