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
    if (( rank==0 ) && ( ((n+1)*(n+1) % numprocs) != 0 )) {
        printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
        exit(1);
    }

    // This sets up various settings to facilitate the
    // MPI communication in a compact manner.
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
    double* localg = init_localg(chunklength, locald->locald);
    double* localu = (double*) calloc(chunklength*chunklength,sizeof(double));
    double* localq = (double*) calloc(chunklength*chunklength,sizeof(double));

    int i;
    double q0, tau, q1, beta;
    MPI_Barrier(mpi_settings->cartcomm);
    double runtime = MPI_Wtime();

    if (numprocs == 1) {
        //// Step 2: calculate q0, then begin the conjugate gradient procedure.
        dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q0);
        for (int iter=0; iter<MAX_ITERS; iter++) {
            // Step 4: q = Ad using a stencil-based product in serial.
            apply_stencil_serial(n, stencil, locald->locald, localq);

            // Steps 5-7: calculate tau, then update u and g.
            dot(chunklength, chunklength, locald->locald, localq, mpi_settings->cartcomm, &tau);
            tau = q0/tau;
            for (i=0; i<chunkarea; ++i) { localu[i] += tau*(locald->locald[i]); }
            for (i=0; i<chunkarea; ++i) { localg[i] += tau*localq[i]; }

            // Steps 8-10: calculate beta, then update d.
            dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q1);
            beta = q1/q0;
            for (i=0; i<chunkarea; ++i) { locald->locald[i] = beta*(locald->locald[i]) - localg[i]; }

            q0 = q1;
        }
    }
    else {
        //// Step 2: calculate q0, then begin the conjugate gradient procedure.
        dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q0);
        for (int iter=0; iter<MAX_ITERS; iter++) {
            // Step 4: q = Ad using a stencil-based product in parallel.
            apply_stencil_parallel(chunklength, stencil, locald, localq, rank, mpi_settings);

            // Steps 5-7: calculate tau, then update u and g.
            dot(chunklength, chunklength, locald->locald, localq, mpi_settings->cartcomm, &tau);
            tau = q0/tau;
            for (i=0; i<chunkarea; ++i) { localu[i] += tau*(locald->locald[i]); }
            for (i=0; i<chunkarea; ++i) { localg[i] += tau*localq[i]; }

            // Steps 8-10: calculate beta, then update d.
            dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q1);
            beta = q1/q0;
            for (i=0; i<chunkarea; ++i) { locald->locald[i] = beta*(locald->locald[i]) - localg[i]; }

            q0 = q1;
        }
    }
    MPI_Barrier(mpi_settings->cartcomm);
    runtime = MPI_Wtime() - runtime;
    double max_runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_settings->cartcomm);
    //print_local2dmesh(chunklength, chunklength, localu, rank, mpi_settings->cartcomm);

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