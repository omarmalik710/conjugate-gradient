#define MAX_ITERS 200
#define UNROLL_FACT 4
#define BLOCK_SIZE 32

typedef struct my_d {
    double* restrict left_pad;
    double* restrict top_pad;
    double* restrict locald;
    double* restrict bottom_pad;
    double* restrict right_pad;
    double* restrict corner_pad;
} d_struct;

typedef struct my_MPI_Settings {
    int cartsize;
    int leftrank;
    int toprank;
    int bottomrank;
    int rightrank;
    int* restrict itags;
    int* restrict jtags;
    MPI_Comm cartcomm;
    MPI_Datatype rowtype;
    MPI_Datatype coltype;
    MPI_Request* restrict irequests;
    MPI_Request* restrict jrequests;
    MPI_Status status;
} MPI_Settings;

void apply_stencil_serial(int n, int* restrict stencil, double* restrict d, double* restrict q);
void apply_stencil_parallel(const int chunklength, int* restrict stencil, d_struct* restrict locald, double* restrict localq, const int myrank, MPI_Settings* restrict mpi_settings);
void exchange_boundaries(const int chunklength, d_struct* restrict locald, const int myrank, MPI_Settings* restrict mpi_settings);
d_struct* init_locald(const int n, const int chunklength, const int myrank, MPI_Settings* restrict mpi_settings);
double* init_localg(const int chunklength, double* restrict d);
void print_local2dmesh(const int rows, const int cols, double* restrict mesh, const int myrank, MPI_Comm cartcomm);
void dot(const int rows, const int cols, double* restrict localv, double* restrict localw, MPI_Comm comm, double* restrict result);
MPI_Settings* init_mpi_settings(int numprocs, int chunklength);
void free_struct_elems(d_struct* restrict locald, MPI_Settings* restrict mpi_settings);