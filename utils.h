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
    double* corner_pad;
} d_struct;

typedef struct my_MPI_Settings {
    int cartsize;
    int* itags;
    int* jtags;
    MPI_Request* irequests;
    MPI_Request* jrequests;
    MPI_Status status;
    MPI_Datatype rowtype;
    MPI_Datatype coltype;
    MPI_Comm cartcomm;
} MPI_Settings;

void apply_stencil_serial(int n, stencil_struct* my_stencil, double* d, double* q);
void apply_stencil_parallel(const int chunklength, stencil_struct* my_stencil, d_struct* locald, double* localq, const int myrank, MPI_Settings* mpi_settings);
void exchange_boundaries(const int chunklength, d_struct* locald, const int myrank, MPI_Settings* mpi_settings);
d_struct* init_locald(const int n, const int chunklength, const int myrank, MPI_Settings* mpi_settings);
double* init_localg(const int chunklength, double* d);
void print_local2dmesh(const int rows, const int cols, double* mesh, const int myrank, MPI_Comm cartcomm);
void dot(const int rows, const int cols, double* localv, double* localw, MPI_Comm comm, double* result);
MPI_Settings* init_mpi_settings(int numprocs, int chunklength);