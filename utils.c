#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

void apply_stencil_serial(const int n, int* restrict stencil, double* restrict d, double* restrict q) {

    double result;
    int i,j;
    int dindexi;
    for (i=1; i<n; ++i) {
        // dindexi iterates over the stencil's rows. It is stored
        // separately to avoid repeated multiplications, which
        // are quite expensive.
        dindexi = (i-1)*(n+1);
        for (j=1; j<n; ++j) {

            // Applying the stencil application on the inner points is the main
            // bottleneck of the code. Replacing the double loops with the 5-line
            // explicit stencil computation will significantly improve performance.
            result = 0;

            result += stencil[0] * d[dindexi + j];
            result += stencil[1] * d[dindexi+(n+1) + (j-1)];
            result += stencil[2] * d[dindexi+(n+1) + j];
            result += stencil[3] * d[dindexi+(n+1) + (j+1)];
            result += stencil[4] * d[dindexi+(n+1)+(n+1) + j];

            q[dindexi+(n+1) + j] = result;
        }
    }
}

void apply_stencil_parallel(const int chunklength, int* restrict stencil, d_struct* restrict locald_struct, double* restrict localq, const int myrank, MPI_Settings* restrict mpi_settings) {

    // Exchange top and/ or bottom padded boundaries
    // using nonblocking communication.
    exchange_boundaries(chunklength, locald_struct, myrank, mpi_settings);

    // Apply stencil on inner points while exchanging
    // the padded boundaries.
    double* locald = locald_struct->locald;
    apply_stencil_serial(chunklength-1, stencil, locald, localq);

    // Wait for receive of padded boundaries from the left, top,
    // bottom, and/or right neighbors to finish before applying the
    // stencil on the outer points.
    if (locald_struct->top_pad != NULL) {
        MPI_Wait((mpi_settings->irequests)+(mpi_settings->toprank),
                 &(mpi_settings->status));
    }
    if (locald_struct->bottom_pad != NULL) {
        MPI_Wait((mpi_settings->irequests)+(mpi_settings->bottomrank),
                 &(mpi_settings->status));
    }

    // Apply stencil on outer points. Use left, top, bottom, and/or
    // right pads when applicable. Non-corner points use 1 of the 4
    // padded arrays. Corner points use 2 of them. Both cases will be
    // explicitly handled here.
    double result;
    int i,j;
    int dindexi;

    //// Non-corner points: first row (i==0).
    if (locald_struct->top_pad != NULL) {
        i = 0;
        dindexi = (i-1)*chunklength;
        for (j=1; j<chunklength-1; ++j) {
            result = 0;

            result += stencil[0] * locald_struct->top_pad[j];
            result += stencil[1] * locald[dindexi+chunklength + (j-1)];
            result += stencil[2] * locald[dindexi+chunklength + j];
            result += stencil[3] * locald[dindexi+chunklength + (j+1)];
            result += stencil[4] * locald[dindexi+chunklength+chunklength + j];

            localq[j] = result;
        }
    }
    //// Non-corner points: last row (i==chunk-1).
    if (locald_struct->bottom_pad != NULL) {
        i = chunklength - 1;
        dindexi = (i-1)*chunklength;
        for (j=1; j<chunklength-1; ++j) {
            result = 0;

            result += stencil[0] * locald[dindexi + j];
            result += stencil[1] * locald[dindexi+chunklength + (j-1)];
            result += stencil[2] * locald[dindexi+chunklength + j];
            result += stencil[3] * locald[dindexi+chunklength + (j+1)];
            result += stencil[4] * locald_struct->bottom_pad[j];

            localq[dindexi+chunklength + j] = result;
        }
    }

    // Wait for receive of padded boundaries from the left
    // and/or right neighbors before applying the stencil on
    // the first and last columns. (Left and right edges are
    // analogous to the top and bottom cases above: just make
    // the swaps i->j and l->m.)
    if (locald_struct->left_pad != NULL) {
        MPI_Wait((mpi_settings->jrequests)+(mpi_settings->leftrank),
                 &(mpi_settings->status));
    }
    if (locald_struct->right_pad != NULL) {
        MPI_Wait((mpi_settings->jrequests)+(mpi_settings->rightrank),
                 &(mpi_settings->status));
    }

    //// Non-corner points: first column (j==0).
    if (locald_struct->left_pad != NULL) {
        j = 0;
        for (i=1; i<chunklength-1; ++i) {
            dindexi = (i-1)*chunklength;
            result = 0;

            result += stencil[0] * locald[dindexi + j];
            result += stencil[1] * locald_struct->left_pad[i];
            result += stencil[2] * locald[dindexi+chunklength + j];
            result += stencil[3] * locald[dindexi+chunklength + j+1];
            result += stencil[4] * locald[dindexi+chunklength+chunklength + j];

            localq[dindexi+chunklength] = result;
        }
    }
    //// Non-corner points: last column (j==chunklength-1).
    if (locald_struct->right_pad != NULL) {
        j = chunklength - 1;
        for (i=1; i<chunklength-1; ++i) {
            dindexi = (i-1)*chunklength;
            result = 0;

            result += stencil[0] * locald[dindexi + j];
            result += stencil[1] * locald[dindexi+chunklength + (j-1)];
            result += stencil[2] * locald[dindexi+chunklength + j];
            result += stencil[3] * locald_struct->right_pad[i];
            result += stencil[4] * locald[dindexi+chunklength+chunklength + j];

            localq[dindexi+chunklength + j] = result;
        }
    }

    // Handle the corner points explicitly.

    //// Corner points: top-left corner (i==0, j==0)
    if (locald_struct->left_pad != NULL && locald_struct->top_pad != NULL) {
        i = 0;
        j = 0;
        dindexi = (i-1)*chunklength;
        result = 0;

        result += stencil[0] * locald_struct->top_pad[j];
        result += stencil[1] * locald_struct->left_pad[i];
        result += stencil[2] * locald[dindexi+chunklength + j];
        result += stencil[3] * locald[dindexi+chunklength + j+1];
        result += stencil[4] * locald[dindexi+chunklength+chunklength + j];

        localq[0] = result;
    }

    //// Corner points: top-right corner (i==0, j==chunklength-1)
    j = chunklength - 1;
    if (locald_struct->top_pad != NULL && locald_struct->right_pad != NULL) {
        i = 0;
        j = chunklength - 1;
        dindexi = (i-1)*chunklength;
        result = 0;

        result += stencil[0] * locald_struct->top_pad[j];
        result += stencil[1] * locald[dindexi+chunklength + (j-1)];
        result += stencil[2] * locald[dindexi+chunklength + j];
        result += stencil[3] * locald_struct->right_pad[i];
        result += stencil[4] * locald[dindexi+chunklength+chunklength + j];

        localq[j] = result;
    }

    //// Corner points: bottom-left corner (i==chunklength-1, j==0)
    if (locald_struct->left_pad != NULL && locald_struct->bottom_pad != NULL) {
        i = chunklength - 1;
        j = 0;
        dindexi = (i-1)*chunklength;
        result = 0;

        result += stencil[0] * locald[dindexi + j];
        result += stencil[1] * locald_struct->left_pad[i];
        result += stencil[2] * locald[dindexi+chunklength + j];
        result += stencil[3] * locald[dindexi+chunklength + (j+1)];
        result += stencil[4] * locald_struct->bottom_pad[j];

        localq[dindexi+chunklength] = result;
    }

    //// Corner points: bottom-right corner (i==chunklength-1, j==chunklength-1)
    if (locald_struct->bottom_pad != NULL && locald_struct->right_pad != NULL) {
        i = chunklength - 1;
        j = chunklength - 1;
        dindexi = (i-1)*chunklength;
        result = 0;

        result += stencil[0] * locald[dindexi + j];
        result += stencil[1] * locald[dindexi+chunklength + (j-1)];
        result += stencil[2] * locald[dindexi+chunklength + j];
        result += stencil[3] * locald_struct->right_pad[i];
        result += stencil[4] * locald_struct->bottom_pad[j];

        localq[dindexi+chunklength + j] = result;
    }
}

void exchange_boundaries(const int chunklength, d_struct* restrict locald_struct, const int myrank, MPI_Settings* restrict mpi_settings) {

    MPI_Comm cartcomm = mpi_settings->cartcomm;
    double *sendbuf, *recvbuf;

    /* HORIZONTAL EXCHANGE */
    int* itags = mpi_settings->itags;
    MPI_Request* irequests = mpi_settings->irequests;
    MPI_Datatype rowtype = mpi_settings->rowtype;

    // Send from / recv to bottom neighbor. First
    // and intermediate rows of procs will do this.
    int bottomrank = mpi_settings->bottomrank;
    if (locald_struct->bottom_pad != NULL) {
        sendbuf = locald_struct->locald + (chunklength-1)*chunklength;
        recvbuf = locald_struct->bottom_pad;
        MPI_Isend(sendbuf, 1, rowtype, bottomrank, itags[myrank], cartcomm, irequests+myrank);
        MPI_Irecv(recvbuf, 1, rowtype, bottomrank, itags[bottomrank], cartcomm, irequests+bottomrank);
    }
    // Send from / recv to top boundary. Last
    // and intermediate rows of procs will do this.
    int toprank = mpi_settings->toprank;
    if (locald_struct->top_pad != NULL) {
        sendbuf = locald_struct->locald;
        recvbuf = locald_struct->top_pad;
        MPI_Isend(sendbuf, 1, rowtype, toprank, itags[myrank], cartcomm, irequests+myrank);
        MPI_Irecv(recvbuf, 1, rowtype, toprank, itags[toprank], cartcomm, irequests+toprank);
    }

    /* VERTICAL EXCHANGE */
    int* jtags = mpi_settings->jtags;
    MPI_Request* jrequests = mpi_settings->jrequests;
    MPI_Datatype coltype = mpi_settings->coltype;

    // Send from / recv to right neighbor. First
    // and intermediate columns of procs will do this.
    int rightrank = mpi_settings->rightrank;
    if (locald_struct->right_pad != NULL) {
        sendbuf = locald_struct->locald + (chunklength-1);
        recvbuf = locald_struct->right_pad;
        MPI_Isend(sendbuf, 1, coltype, rightrank, jtags[myrank], cartcomm, jrequests+myrank);
        MPI_Irecv(recvbuf, 1, rowtype, rightrank, jtags[rightrank], cartcomm, jrequests+rightrank);
    }
    // Send from / recv to left boundary. Last
    // and intermediate columns of procs will do this.
    int leftrank = mpi_settings->leftrank;
    if (locald_struct->left_pad != NULL) {
        sendbuf = locald_struct->locald;
        recvbuf = locald_struct->left_pad;
        MPI_Isend(sendbuf, 1, coltype, leftrank, jtags[myrank], cartcomm, jrequests+myrank);
        MPI_Irecv(recvbuf, 1, rowtype, leftrank, jtags[leftrank], cartcomm, jrequests+leftrank);
    }

}

d_struct* init_locald(const int n, const int chunklength, const int myrank, MPI_Settings* restrict mpi_settings) {
    d_struct* restrict locald_struct = (d_struct*) malloc(sizeof(d_struct));
    locald_struct->locald = (double*) malloc(chunklength*chunklength*sizeof(double));

    MPI_Comm cartcomm = mpi_settings->cartcomm;
    const int cartsize = mpi_settings->cartsize;
    int coords[2];
    MPI_Cart_coords(cartcomm, myrank, 2, coords);
    const int carti = coords[0];
    const int cartj = coords[1];

    // Initialize top and bottom pads in cartesian topology. First and
    // last rows of procs are padded in one direction, while inner rows
    // are padded in two.
    if ( carti==0 ) { // First row of processors.
        locald_struct->top_pad = NULL;
        locald_struct->bottom_pad = (double*) calloc(chunklength,sizeof(double));
    }
    else if ( carti==(cartsize-1) ) { // Last row of processors.
        locald_struct->top_pad = (double*) calloc(chunklength,sizeof(double));
        locald_struct->bottom_pad = NULL;
    }
    else { // Intermediate processors.
        locald_struct->top_pad = (double*) calloc(chunklength,sizeof(double));
        locald_struct->bottom_pad = (double*) calloc(chunklength,sizeof(double));
    }

    // Initialize left and right pads in cartesian topology. First and
    // last columns of procs are padded in one direction, while inner
    // columns are padded in two.
    if ( cartj==0 ) { // First column of processors.
        locald_struct->left_pad = NULL;
        locald_struct->right_pad = (double*) calloc(chunklength,sizeof(double));
    }
    else if ( cartj==(cartsize-1) ) { // Last column of processors.
        locald_struct->left_pad = (double*) calloc(chunklength,sizeof(double));
        locald_struct->right_pad = NULL;
    }
    else { // Intermediate columns of processors.
        locald_struct->left_pad = (double*) calloc(chunklength,sizeof(double));
        locald_struct->right_pad = (double*) calloc(chunklength,sizeof(double));
    }

    // Initialize local d. No need to optimize this, since it's
    // only done once (at the start of the program).
    int i,j;
    double xi, yj;
    const double h = (double) 1 / n;
    for (i=0; i<chunklength; ++i) {
        xi = (i+carti*chunklength)*h;
        for (j=0; j<chunklength; ++j) {
            yj = (j+cartj*chunklength)*h;
            // Account for boundary conditions.
            if ( (carti==0 && i==0) || (carti==(cartsize-1) && i==(chunklength-1)) ||
                 (cartj==0 && j==0) || (cartj==(cartsize-1) && j==(chunklength-1)) ) {
                locald_struct->locald[i*chunklength+j] = 0.0;
            }
            else {
                locald_struct->locald[i*chunklength+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }
        }
    }

    return locald_struct;
}

double* init_localg(const int chunklength, double* restrict locald) {
    double* restrict localg = (double*) malloc(chunklength*chunklength*sizeof(double));
    for (int i=0; i<chunklength; ++i) {
        for (int j=0; j<chunklength; ++j) {
            localg[i*chunklength+j] = -locald[i*chunklength+j];
        }
    }
    return localg;
}

void print_local2dmesh(const int rows, const int cols, double* restrict mesh, const int myrank, MPI_Comm cartcomm) {
    int coords[2];
    MPI_Cart_coords(cartcomm, myrank, 2, coords);
    const int carti = coords[0];
    const int cartj = coords[1];

    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            if (mesh != NULL) {
                printf("([%d (%d,%d)] %lf) ", myrank, carti, cartj, mesh[i*cols+j]);
            }
        }
        putchar('\n');
    }
}

void dot(const int rows, const int cols, double* restrict localv, double* restrict localw, MPI_Comm comm, double* restrict result) {
    int N = rows*cols;
    int i, istart;
    int iblock;
    int numblocks = N/BLOCK_SIZE;
    int blockremain = N%BLOCK_SIZE;
    double localsum = 0.0;
    for (i=0; i<blockremain; ++i) { localsum += localv[i]*localw[i]; }
    for (iblock=0; iblock<numblocks; iblock++) {
        istart = blockremain + iblock*BLOCK_SIZE;
        for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
            localsum += localv[i]*localw[i];
            localsum += localv[i+1]*localw[i+1];
            localsum += localv[i+2]*localw[i+2];
            localsum += localv[i+3]*localw[i+3];
        }
    }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}

MPI_Settings* init_mpi_settings(int numprocs, int chunklength) {
    MPI_Settings* restrict mpi_settings = (MPI_Settings*) malloc(sizeof(MPI_Settings));

    // All matrices are stored row-wise, so row datatype should be contiguous.
    MPI_Type_contiguous(chunklength, MPI_DOUBLE, &(mpi_settings->rowtype));
    MPI_Type_commit(&(mpi_settings->rowtype));

    // Because all matrices are stored row-wise, the column data type
    // must be a vector that strides over every chunklength elements.
    MPI_Type_vector(chunklength, 1, chunklength, MPI_DOUBLE, &(mpi_settings->coltype));
    MPI_Type_commit(&(mpi_settings->coltype));

    // Initialize requests and tags for data exchange in i- and j-
     // directions. i-direction is up/down and j-direction is left/right.
    mpi_settings->irequests = (MPI_Request*) malloc(numprocs*sizeof(MPI_Request));
    mpi_settings->jrequests = (MPI_Request*) malloc(numprocs*sizeof(MPI_Request));
    mpi_settings->itags = (int*) malloc(numprocs*sizeof(int));
    mpi_settings->jtags = (int*) malloc(numprocs*sizeof(int));
    for (int i=0; i<numprocs; ++i) {
        mpi_settings->itags[i] = i;
        mpi_settings->jtags[i] = i;
    }

    // Set up a 2d cartesian topology of procs to facilitate
    // data exchange between them.
    mpi_settings->cartsize = (int) sqrt(numprocs);
    int dim[2], period[2], reorder;
    dim[0] = dim[1] = (mpi_settings->cartsize);
    period[0] = period[1] = 0;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &(mpi_settings->cartcomm));

    // Get each processors neighbors if present.
    MPI_Cart_shift(mpi_settings->cartcomm, 0, 1, &(mpi_settings->toprank), &(mpi_settings->bottomrank));
    MPI_Cart_shift(mpi_settings->cartcomm, 1, 1, &(mpi_settings->leftrank), &(mpi_settings->rightrank));

    return mpi_settings;
}

void free_struct_elems(d_struct* restrict locald, MPI_Settings* restrict mpi_settings) {

    free(locald->left_pad);
    free(locald->top_pad);
    free(locald->locald);
    free(locald->bottom_pad);
    free(locald->right_pad);

    free(mpi_settings->itags);
    free(mpi_settings->jtags);
    free(mpi_settings->irequests);
    free(mpi_settings->jrequests);
}