#define MAX_ITERS 200
typedef struct my_stencil {
    int size;
    int extent;
    int* stencil;
} stencil_struct;

typedef struct my_d {
    double* top_pad;
    double* locald;
    double* bottom_pad;
} d_struct;

//double* init_locald(int n, int rank, int numprocs, int chunk);
d_struct* init_locald(int n, int rank, int numprocs, int chunk);
void exchange_boundaries(int n, double* d, int rank, int numprocs, int chunk);
double* init_localg(int n, double* d, int rank, int chunk);
void print_local2dmesh(int rows, int cols, double* mesh, int rank);
void apply_stencil(int n, stencil_struct my_stencil, double* src, double* dest);
void dot(int rows, int cols, double* localv, double* localw,
        int rank, MPI_Comm comm, double* result);