#define MAX_ITERS 200
typedef struct my_stencil {
    int size;
    int extent;
    int* stencil;
} stencil_struct;

double* init_locald(int n, int rank, int numprocs, int chunk);
void exchange_boundaries(int n, double* d, int rank, int numprocs, int chunk);
double* init_localg(int n, double* d, int rank, int chunk);
void print_local2dmesh(int n, double* mesh, int rank, int numprocs, int chunk);
void apply_stencil(int n, stencil_struct my_stencil, double* src, double* dest);
void dot(int n, double* localv, double* localw,
        int rank, int chunk, MPI_Comm comm,
        double* result);