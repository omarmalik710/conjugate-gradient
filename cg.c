#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv) {

    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sqrt_numprocs = sqrt(numprocs);
    if (( rank==0 ) && ( (int)sqrt_numprocs*sqrt_numprocs != numprocs )) {
        printf("[ERROR] Number of processors '%d' is not a perfect square.\n", numprocs);
        exit(1);
    }

    int n = atoi(argv[1]);
    int chunkarea = (n+1)*(n+1) / numprocs;
    int chunklength = sqrt(chunkarea);
    int localN = chunklength*chunklength;
    if (( rank==0 ) && ( ((n+1)*(n+1) % numprocs) != 0 )) {
        printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
        exit(1);
    }

    MPI_Settings* mpi_settings = init_mpi_settings(numprocs, chunklength);

    // Initialize stencil.
    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct* stencil = (stencil_struct*) malloc(sizeof(stencil));
    stencil->size = 3;
    stencil->extent = stencil->size/2;
    stencil->stencil = (int*) malloc(stencil->size*stencil->size*sizeof(int));
    memcpy(stencil->stencil, my_stencil, sizeof(my_stencil));

    // Initialize local arrays.
    d_struct* locald = init_locald(n, chunklength, rank, mpi_settings);

    //print_local2dmesh(1, chunklength, locald->left_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->top_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(chunklength, chunklength, locald->locald, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->bottom_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->right_pad, rank, mpi_settings->cartcomm);
    //exchange_boundaries(chunklength, locald, rank, mpi_settings);
    //print_local2dmesh(1, chunklength, locald->left_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->top_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(chunklength, chunklength, locald->locald, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->bottom_pad, rank, mpi_settings->cartcomm);
    //print_local2dmesh(1, chunklength, locald->right_pad, rank, mpi_settings->cartcomm);

    double* localg = init_localg(chunklength, locald->locald);
    //print_local2dmesh(chunklength, chunklength, localg, rank, cartcomm);
    double* localu = (double*) calloc(chunklength*chunklength,sizeof(double));
    double* localq = (double*) calloc(chunklength*chunklength,sizeof(double));

    int i;
    double q0, tau, q1, beta;
    MPI_Barrier(MPI_COMM_WORLD);
    double runtime = MPI_Wtime();

    //// Step 2: q0 = dot(g,g)
    dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q0);
    //printf("[INFO] q0 = %.16lf\n", q0);
    for (int iter=0; iter<MAX_ITERS; iter++) {
        //printf("[RANK %d, Iter %d]\n", rank, iter);

        // Step 4: q = Ad. Exchange boundaries first
        // before applying the stencil.
        //printf("[RANK %d, Iter %d]\n", rank, iter);
        apply_stencil(chunklength, stencil, locald, localq, rank, mpi_settings);

        // Step 5: tau = q0/dot(d,q)
        dot(chunklength, chunklength, locald->locald, localq, mpi_settings->cartcomm, &tau);
        tau = q0/tau;

        // Step 6: u = u + tau*d
        for (i=0; i<localN; ++i) { localu[i] += tau*(locald->locald[i]); }

        // Step 7: g = g + tau*q
        for (i=0; i<localN; ++i) { localg[i] += tau*localq[i]; }

        // Step 8: q1 = dot(g,g)
        dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q1);
        beta = q1/q0;

        // Step 10: d = -g + beta*d
        for (i=0; i<localN; ++i) { locald->locald[i] = beta*(locald->locald[i]) - localg[i]; }
        q0 = q1;
    }
    runtime = MPI_Wtime() - runtime;
    double max_runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_settings->cartcomm);

    // Output: norm(g) = sqrt( dot(g,g) )
    double norm_g;
    dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &norm_g);
    norm_g = sqrt(norm_g);
    if (rank==0) {
        printf("[INFO] norm_g = %.16lf\n", norm_g);
        printf("%.16lf\n", max_runtime);
    }

    free(localu);
    free(localg);
    free(localq);
    free(locald);
    free(stencil);
    free(mpi_settings);
    MPI_Finalize();

    return 0;
}