#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char** argv) {

    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dim[2], period[2], reorder;
    double sqrt_numprocs = sqrt(numprocs);
    if ( (rank==0) && ((int)sqrt_numprocs*sqrt_numprocs != numprocs) ) {
        printf("[ERROR] Number of processors '%d' is not a perfect square.\n", numprocs);
        exit(1);
    }

    // Set up cartesian topology of procs.
    MPI_Comm cartcomm;
    int cartsize = (int) sqrt_numprocs;
    dim[0] = dim[1] = cartsize;
    period[0] = period[1] = 0;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartcomm);

    int n = atoi(argv[1]);
    int chunkarea = (n+1)*(n+1) / numprocs;
    int chunklength = sqrt(chunkarea);
    int localN = chunklength*chunklength;
    if (( rank==0 ) && ( ((n+1)*(n+1) % numprocs) != 0 )) {
        printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
        exit(1);
    }

    d_struct* locald = init_locald(n, chunklength, rank, cartcomm, cartsize);
    //print_local2dmesh(chunklength, chunklength, locald->locald, rank, cartcomm);

    int left, top, bottom, right;
    //MPI_Cart_shift(cartcomm, 0, 1, &top, &bottom);
    //printf("[INFO] top %d : me %d : bottom %d\n", top, rank, bottom);

    MPI_Cart_shift(cartcomm, 1, 1, &left, &right);
    printf("[INFO] left %d : me %d : right %d\n", left, rank, right);
    MPI_Finalize();

    return 0;
}