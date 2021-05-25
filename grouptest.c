#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char** argv) {

    int wrank, wnumprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &wnumprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    double sqrt_numprocs = sqrt(wnumprocs);
    if ( (wrank==0) && ((int)sqrt_numprocs*sqrt_numprocs != wnumprocs) ) {
        printf("[ERROR] Number of processors '%d' is not a perfect square.\n", wnumprocs);
        exit(1);
    }
    int color = wrank / (int) sqrt_numprocs;
    int key = wrank % (int) sqrt_numprocs;
    printf("[RANK %d] color = %d, key = %d\n", wrank, color, key);

    int rrank, rnumprocs;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &row_comm);
    MPI_Comm_size(row_comm, &rnumprocs);
    MPI_Comm_rank(row_comm, &rrank);
    printf("[WRANK %d, RRANK %d]\n", wrank, rrank);

    int crank, cnumprocs;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &col_comm);
    MPI_Comm_size(col_comm, &cnumprocs);
    MPI_Comm_rank(col_comm, &crank);
    printf("[WRANK %d, CRANK %d]\n", wrank, crank);

    //int n = atoi(argv[1]);
    //int chunk = (n+1) / numprocs;
    //int localN = chunk*(n+1);
    //if ( (rank==0) && ((n+1)%numprocs != 0) ) {
    //    printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
    //    exit(1);
    //}

    MPI_Finalize();

    return 0;
}