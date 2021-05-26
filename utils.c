#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

void apply_stencil_serial(const int n, stencil_struct* my_stencil, double* d, double* q) {
    const int stencil_size = my_stencil->size;
    const int extent = my_stencil->extent;
    int* stencil = my_stencil->stencil;

    double result;
    int i,j,l,m;
    int index;
    for (i=extent; i<(n+1)-extent; ++i) {
        for (j=extent; j<(n+1)-extent; ++j) {

            // Apply stencil on each inner i,j point in the mesh.
            result = 0;
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * d[index];
                }
            }
            q[i*(n+1)+j] = result;
        }
    }
}

void apply_stencil_parallel(const int chunklength, stencil_struct* my_stencil, d_struct* locald, double* localq, const int myrank, MPI_Settings* mpi_settings) {
    const int stencil_size = my_stencil->size;
    const int extent = my_stencil->extent;
    int* stencil = my_stencil->stencil;

    // Exchange top and/ or bottom padded boundaries
    // using nonblocking communication.
    exchange_boundaries(chunklength, locald, myrank, mpi_settings);

    // Apply stencil on inner points while exchanging
    // the padded boundaries above.
    double result;
    int i,j,l,m;
    int index;
    for (i=extent; i<chunklength-extent; ++i) {
        for (j=extent; j<chunklength-extent; ++j) {

            result = 0;
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*chunklength + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            localq[i*chunklength+j] = result;
        }
    }

    // Apply stencil on outer points. Use left, top, bottom, and/or
    // right pads when applicable. Non-corner points use 1 of the 4
    // padded arrays. Corner points use 2 of them. Both cases will be
    // explicitly handled here.

    // Wait for receive of padded boundaries from the top
    // and/or bottom neighbors to finish before applying the
    // stencil on the first and last rows.
    if (locald->top_pad != NULL) {
        MPI_Wait((mpi_settings->irequests)+(mpi_settings->toprank),
                 &(mpi_settings->status));
    }
    if (locald->bottom_pad != NULL) {
        MPI_Wait((mpi_settings->irequests)+(mpi_settings->bottomrank),
                 &(mpi_settings->status));
    }

    //// Non-corner points: first row (i==0).
    if (locald->top_pad != NULL) {
        for (j=extent; j<chunklength-extent; ++j) {
            result = 0;
            // Handle left, bottom, and right neighbors as usual.
            for (l=1; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (-extent + l)*chunklength + (j - extent + m);
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
    //// Non-corner points: last row (i==chunk-extent).
    if (locald->bottom_pad != NULL) {
        for (j=extent; j<chunklength-extent; ++j) {
            result = 0;
            // Handle left, top, and right neighbors as usual.
            for (l=0; l<stencil_size-1; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*chunklength + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            // Use bottom_pad for the bottom neighbors (l==stencil_size-1).
            for (m=0; m<stencil_size; ++m) {
                index = j - extent + m;
                result += stencil[l*stencil_size+m] * locald->bottom_pad[index];
            }
            localq[i*chunklength+j] = result;
        }
    }

    // Wait for receive of padded boundaries from the left
    // and/or right neighbors before applying the stencil on
    // the first and last columns. (Left and right edges are
    // analogous to the top and bottom cases above: just make
    // the swaps i->j and l->m.)
    if (locald->left_pad != NULL) {
        MPI_Wait((mpi_settings->jrequests)+(mpi_settings->leftrank),
                 &(mpi_settings->status));
    }
    if (locald->right_pad != NULL) {
        MPI_Wait((mpi_settings->jrequests)+(mpi_settings->rightrank),
                 &(mpi_settings->status));
    }

    //// Non-corner points: first column (j==0).
    if (locald->left_pad != NULL) {
        for (i=extent; i<chunklength-extent; ++i) {
            result = 0;
            // Handle top, bottom, and right neighbors as usual.
            for (l=0; l<stencil_size; ++l) {
                for (m=1; m<stencil_size; ++m) {
                    index = (i - extent + l)*chunklength + (-extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            // Use left_pad for the left neighbors (m==0).
            for (l=0; l<stencil_size; ++l) {
                index = i - extent + l;
                result += stencil[l*stencil_size] * locald->left_pad[index];
            }
            localq[i*chunklength] = result;
        }
    }
    //// Non-corner points: last column (j==chunklength-extent).
    if (locald->right_pad != NULL) {
        for (i=extent; i<chunklength-extent; ++i) {
            result = 0;
            // Handle left, top, and bottom neighbors as usual.
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size-1; ++m) {
                    index = (i - extent + l)*chunklength + (j - extent + m);
                    result += stencil[l*stencil_size+m] * locald->locald[index];
                }
            }
            // Use right_pad for the right neighbors (m==stencil_size-1).
            for (l=0; l<stencil_size; ++l) {
                index = i - extent + l;
                result += stencil[l*stencil_size+m] * locald->right_pad[index];
            }
            localq[i*chunklength+j] = result;
        }
    }

    // Handle the corner points explicitly.

    //// Corner points: top-left corner (i==0, j==0)
    if (locald->left_pad != NULL && locald->top_pad != NULL) {
        result = 0;
        // Handle right and bottom neighbors as usual.
        for (l=1; l<stencil_size; ++l) {
            for (m=1; m<stencil_size; ++m) {
                index = (-extent + l)*chunklength + (-extent + m);
                result += stencil[l*stencil_size+m] * locald->locald[index];
            }
        }

        // Use left_pad and top_pad for the left and right neighbors,
        // but iterate separately and exclude the remaining zero
        // elements in the stencil. (Accounting for the corner elements
        // in the stencil is too tedious; thankfully they are zero!)
        //// l==1 and m==0 case (stencil[1][0] = -1)
        index = 0;
        result += stencil[stencil_size] * locald->left_pad[index];
        //// l==0 and m==1 case (stencil[0][1] = -1)
        index = 0 ;
        result += stencil[1] * locald->top_pad[index];
        localq[0] = result;
    }

    //// Corner points: top-right corner (i==0, j==chunklength-extent)
    j = chunklength - extent;
    if (locald->top_pad != NULL && locald->right_pad != NULL) {
        result = 0;
        // Handle left and bottom neighbors as usual.
        for (l=1; l<stencil_size; ++l) {
            for (m=0; m<stencil_size-1; ++m) {
                index = (-extent + l)*chunklength + (j - extent + m);
                result += stencil[l*stencil_size+m] * locald->locald[index];
            }
        }

        // Use top_pad and right_pad for the left and right neighbors,
        // analogously to the left_pad and top_pad case above.
        //// l==0 and m==1 case (stencil[0][1] = -1)
        m = 1;
        index = chunklength - extent ;
        result += stencil[m] * locald->top_pad[index];
        //// l==1 and m==stencil_size-1 case (stencil[1][stencil_size-1] = -1)
        m = stencil_size - 1;
        index = 0;
        result += stencil[stencil_size+m] * locald->right_pad[index];
        localq[j] = result;
    }

    //// Corner points: bottom-left corner (i==chunklength-extent, j==0)
    i = chunklength - extent;
    if (locald->left_pad != NULL && locald->bottom_pad != NULL) {
        result = 0;
        // Handle right and bottom neighbors as usual.
        for (l=0; l<stencil_size-1; ++l) {
            for (m=1; m<stencil_size; ++m) {
                index = (i - extent + l)*chunklength + (-extent + m);
                result += stencil[l*stencil_size+m] * locald->locald[index];
            }
        }

        // Use left_pad and bottom_pad for the left and bottom neighbors,
        // analogous to the left_pad and top_pad case above.
        //// l==1 and m==0 case (stencil[1][0] = -1)
        index = chunklength - extent;
        result += stencil[stencil_size] * locald->left_pad[index];
        //// l==stencil_size-1 and m==1 case (stencil[stencil_size-1][1] = -1)
        l = stencil_size - 1;
        m = 1;
        index = 0;
        result += stencil[l*stencil_size+m] * locald->bottom_pad[index];
        localq[i*chunklength] = result;
    }

    //// Corner points: bottom-right corner (i==chunklength-extent, j==chunklength-extent)
    i = chunklength - extent;
    j = chunklength - extent;
    if (locald->bottom_pad != NULL && locald->right_pad != NULL) {
        result = 0;
        // Handle right and bottom neighbors as usual.
        for (l=0; l<stencil_size-1; ++l) {
            for (m=0; m<stencil_size-1; ++m) {
                index = (i - extent + l)*chunklength + (j - extent + m);
                result += stencil[l*stencil_size+m] * locald->locald[index];
            }
        }

        // Use bottom_pad and right_pad for the right and bottom neighbors,
        // analogous to the left_pad and top_pad case above.
        //// l==stencil_size-1 and m==1 case (stencil[stencil_size-1][1] = -1)
        l = stencil_size - 1;
        m = 1;
        index = chunklength - extent;
        result += stencil[l*stencil_size+m] * locald->bottom_pad[index];
        //// l==1 and m==stencil_size-1 case (stencil[1][stencil_size-1] = -1)
        m = stencil_size - 1;
        index = chunklength - extent;
        result += stencil[stencil_size+m] * locald->right_pad[index];
        localq[i*chunklength+j] = result;
    }
}

void exchange_boundaries(const int chunklength, d_struct* locald, const int myrank, MPI_Settings* mpi_settings) {

    double *sendbuf, *recvbuf;
    // Send from / recv to bottom neighbor. First
    // and intermediate rows of procs will do this.
    int toprank, bottomrank;
    MPI_Cart_shift(mpi_settings->cartcomm, 0, 1, &toprank, &bottomrank);
    if (locald->bottom_pad != NULL) {
        sendbuf = locald->locald + (chunklength-1)*chunklength;
        recvbuf = locald->bottom_pad;
        MPI_Isend(sendbuf, 1, mpi_settings->rowtype, bottomrank,
                  mpi_settings->itags[myrank], mpi_settings->cartcomm,
                  (mpi_settings->irequests)+myrank);
        MPI_Irecv(recvbuf, 1, mpi_settings->rowtype, bottomrank,
                  mpi_settings->itags[bottomrank], mpi_settings->cartcomm,
                  (mpi_settings->irequests)+bottomrank);
    }
    // Send from / recv to top boundary. Last
    // and intermediate rows of procs will do this.
    if (locald->top_pad != NULL) {
        sendbuf = locald->locald;
        recvbuf = locald->top_pad;
        MPI_Isend(sendbuf, 1, mpi_settings->rowtype, toprank,
                  mpi_settings->itags[myrank], mpi_settings->cartcomm,
                  (mpi_settings->irequests)+myrank);
        MPI_Irecv(recvbuf, 1, mpi_settings->rowtype, toprank,
                  mpi_settings->itags[toprank], mpi_settings->cartcomm,
                  (mpi_settings->irequests)+toprank);
    }

    // Send from / recv to left neighbor. First
    // and intermediate columns of procs will do this.
    int leftrank, rightrank;
    MPI_Cart_shift(mpi_settings->cartcomm, 1, 1, &leftrank, &rightrank);
    if (locald->right_pad != NULL) {
        sendbuf = locald->locald + (chunklength-1);
        recvbuf = locald->right_pad;
        MPI_Isend(sendbuf, 1, mpi_settings->coltype, rightrank,
                  mpi_settings->jtags[myrank], mpi_settings->cartcomm,
                  (mpi_settings->jrequests)+myrank);
        MPI_Irecv(recvbuf, 1, mpi_settings->rowtype, rightrank,
                  mpi_settings->jtags[rightrank], mpi_settings->cartcomm,
                  (mpi_settings->jrequests)+rightrank);
    }
    // Send from / recv to left boundary. Last
    // and intermediate columns of procs will do this.
    if (locald->left_pad != NULL) {
        sendbuf = locald->locald;
        recvbuf = locald->left_pad;
        MPI_Isend(sendbuf, 1, mpi_settings->coltype, leftrank,
                  mpi_settings->jtags[myrank], mpi_settings->cartcomm,
                  (mpi_settings->jrequests)+myrank);
        MPI_Irecv(recvbuf, 1, mpi_settings->rowtype, leftrank,
                  mpi_settings->jtags[leftrank], mpi_settings->cartcomm,
                  (mpi_settings->jrequests)+leftrank);
    }

}

d_struct* init_locald(const int n, const int chunklength, const int myrank, MPI_Settings* mpi_settings) {
    d_struct* locald = (d_struct*) malloc(sizeof(d_struct));
    locald->locald = (double*) malloc(chunklength*chunklength*sizeof(double));

    MPI_Comm cartcomm = mpi_settings->cartcomm;
    const int cartsize = mpi_settings->cartsize;
    int coords[2];
    MPI_Cart_coords(cartcomm, myrank, 2, coords);
    const int carti = coords[0];
    const int cartj = coords[1];

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
    const double h = (double) 1 / n;
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

double* init_localg(const int chunklength, double* d) {
    double* g = (double*) malloc(chunklength*chunklength*sizeof(double));
    for (int i=0; i<chunklength; ++i) {
        for (int j=0; j<chunklength; ++j) {
            g[i*chunklength+j] = -d[i*chunklength+j];
        }
    }
    return g;
}

void print_local2dmesh(const int rows, const int cols, double* mesh, const int myrank, MPI_Comm cartcomm) {
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

void dot(const int rows, const int cols, double* localv, double* localw, MPI_Comm comm, double* result) {
    double localsum = 0.0;
    for (int i=0; i<rows*cols; ++i) { localsum += localv[i]*localw[i]; }
    MPI_Allreduce(&localsum, result, 1, MPI_DOUBLE, MPI_SUM, comm);
}

MPI_Settings* init_mpi_settings(int numprocs, int chunklength) {
    MPI_Settings* mpi_settings = (MPI_Settings*) malloc(sizeof(MPI_Settings));

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

void free_struct_elems(stencil_struct* stencil, d_struct* locald, MPI_Settings* mpi_settings) {
    free(stencil->stencil);

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