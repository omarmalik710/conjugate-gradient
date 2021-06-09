#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv) {

    if (BLOCK_SIZE%UNROLL_FACT != 0) {
        printf("[ERROR] Cache block size '%d' is not disivible by loop unroll factor '%d'.\n",
                BLOCK_SIZE, UNROLL_FACT);
        exit(1);
    }

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
    MPI_Settings* restrict mpi_settings = init_mpi_settings(numprocs, chunklength);

    // Initialize stencil.
    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct* restrict stencil = (stencil_struct*) malloc(sizeof(stencil));
    stencil->size = 3;
    stencil->extent = stencil->size/2;
    stencil->stencil = (int*) malloc(stencil->size*stencil->size*sizeof(int));
    memcpy(stencil->stencil, my_stencil, sizeof(my_stencil));

    // Initialize local arrays.
    d_struct* locald_struct = init_locald(n, chunklength, rank, mpi_settings);
    double* restrict locald = locald_struct->locald;
    double* restrict localg = init_localg(chunklength, locald);
    double* restrict localu = (double*) calloc(chunklength*chunklength,sizeof(double));
    double* restrict localq = (double*) calloc(chunklength*chunklength,sizeof(double));

    int i, istart;
    int iblock;
    int numblocks = chunkarea/BLOCK_SIZE;
    int blockremain = chunkarea%BLOCK_SIZE;
    double q0, tau, q1, beta;
    MPI_Barrier(mpi_settings->cartcomm);
    double runtime = MPI_Wtime();

    if (numprocs == 1) { // If only 1 processer is available.

        //// Step 2: calculate q0, then begin the conjugate gradient procedure.
        dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q0);
        for (int iter=0; iter<MAX_ITERS; iter++) {
            // Step 4: q = Ad using a stencil-based product in serial.
            apply_stencil_serial(n, stencil, locald, localq);

            // Steps 5-7: calculate tau, then update u and g with cache blocking
            // and loop unrolling.
            dot(chunklength, chunklength, locald, localq, mpi_settings->cartcomm, &tau);
            tau = q0/tau;
            for (i=0; i<blockremain; ++i) { localu[i] += tau*locald[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    localu[i] += tau*locald[i];
                    localu[i+1] += tau*locald[i+1];
                    localu[i+2] += tau*locald[i+2];
                    localu[i+3] += tau*locald[i+3];
                }
            }
            for (i=0; i<blockremain; ++i) { localg[i] += tau*localq[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    localg[i] += tau*localq[i];
                    localg[i+1] += tau*localq[i+1];
                    localg[i+2] += tau*localq[i+2];
                    localg[i+3] += tau*localq[i+3];
                }
            }

            // Steps 8-10: calculate beta, then update d with cache blocking
            // and loop unrolling.
            dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q1);
            beta = q1/q0;
            for (i=0; i<blockremain; ++i) { locald[i] = beta*locald[i] - localg[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    locald[i] = beta*locald[i] - localg[i];
                    locald[i+1] = beta*locald[i+1] - localg[i+1];
                    locald[i+2] = beta*locald[i+2] - localg[i+2];
                    locald[i+3] = beta*locald[i+3] - localg[i+3];
                }
            }

            q0 = q1;
        }
    }
    else { // For more than 1 processor.

        //// Step 2: calculate q0, then begin the conjugate gradient procedure.
        dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q0);
        for (int iter=0; iter<MAX_ITERS; iter++) {
            // Step 4: q = Ad using a stencil-based product in parallel.
            apply_stencil_parallel(chunklength, stencil, locald_struct, localq, rank, mpi_settings);

            // Steps 5-7: calculate tau, then update u and g with cache blocking
            // and loop unrolling.
            dot(chunklength, chunklength, locald, localq, mpi_settings->cartcomm, &tau);
            tau = q0/tau;
            for (i=0; i<blockremain; ++i) { localu[i] += tau*locald[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    localu[i] += tau*locald[i];
                    localu[i+1] += tau*locald[i+1];
                    localu[i+2] += tau*locald[i+2];
                    localu[i+3] += tau*locald[i+3];
                }
            }
            for (i=0; i<blockremain; ++i) { localg[i] += tau*localq[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    localg[i] += tau*localq[i];
                    localg[i+1] += tau*localq[i+1];
                    localg[i+2] += tau*localq[i+2];
                    localg[i+3] += tau*localq[i+3];
                }
            }

            // Steps 8-10: calculate beta, then update d with cache blocking
            // and loop unrolling.
            dot(chunklength, chunklength, localg, localg, mpi_settings->cartcomm, &q1);
            beta = q1/q0;
            for (i=0; i<blockremain; ++i) { locald[i] = beta*locald[i] - localg[i]; }
            for (iblock=0; iblock<numblocks; iblock++) {
                istart = blockremain + iblock*BLOCK_SIZE;
                for (i=istart; i<(istart+BLOCK_SIZE); i+=UNROLL_FACT) {
                    locald[i] = beta*locald[i] - localg[i];
                    locald[i+1] = beta*locald[i+1] - localg[i+1];
                    locald[i+2] = beta*locald[i+2] - localg[i+2];
                    locald[i+3] = beta*locald[i+3] - localg[i+3];
                }
            }

            q0 = q1;
        }
    }
    MPI_Barrier(mpi_settings->cartcomm);
    runtime = MPI_Wtime() - runtime;
    double max_runtime;
    MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_settings->cartcomm);
    //print_local2dmesh(chunklength, chunklength, localu, rank, mpi_settings->cartcomm);

    // Output: norm(g) = sqrt( dot(g,g) )
    if (rank==0) {
        printf("[INFO] norm_g^2 = %.16lf\n", q1);
        printf("%.16lf\n", max_runtime);
    }

    free(localu);
    free(localg);
    free(localq);
    free_struct_elems(stencil, locald_struct, mpi_settings);
    free(locald_struct);
    free(stencil);
    free(mpi_settings);
    MPI_Finalize();

    return 0;
}