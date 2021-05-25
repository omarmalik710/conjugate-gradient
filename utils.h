#define MAX_ITERS 200
typedef struct my_stencil {
    int size;
    int extent;
    int* stencil;
} stencil_struct;

typedef struct my_d {
    double* left_pad;
    double* top_pad;
    double* locald;
    double* bottom_pad;
    double* right_pad;
} d_struct;

d_struct* init_locald(int n, int chunklength, int wrank, MPI_Comm cartcomm, int cartsize);
void exchange_boundaries(int n, d_struct* locald, int rank, int numprocs, int chunk, MPI_Request* requests);
double* init_localg(int n, double* d, int rank, int chunk);
void print_local2dmesh(int rows, int cols, double* mesh, int wrank, MPI_Comm cartcomm);
void apply_stencil(int n, stencil_struct* my_stencil, d_struct* locald, double* localq, int rank, int numprocs, int chunk);
void dot(int rows, int cols, double* localv, double* localw,
        int rank, MPI_Comm comm, double* result);