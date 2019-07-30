#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define num_row 500
#define num_col 500       /* number of rows and columns in matrix */

double matrix_a[num_row][num_col], matrix_b[num_row][num_col], matrix_c[num_row][num_col];

/* row-major order */
#define ind(i,j) ((i)*(MAX_COL)+(j))

int ind_f(int i, int j, int MAX_COL)
{
    return ind(i, j);
}

void print_matrix(double matrix[], int MAX_ROW, int MAX_COLUMN);

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;

    int i, j, k, numworkers, rows, source, dest, offset;
    double t1, t2;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    numworkers = size - 1;

    if (rank == 0) {
        /* ------------- Master -------------- */
        for (i=0; i<num_row; i++) {
            for (j=0; j<num_col; j++) {
                matrix_a[i][j]= 1.0;
                matrix_b[i][j]= 2.0;
            }
        }
        /* send matrix data to the worker tasks */
        rows = num_row/numworkers;
        offset = 0;
        t1 = MPI_Wtime();   /* take time */
        for (dest=1; dest<=numworkers; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_a[offset][0], rows*num_col, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
            MPI_Send(&matrix_b, num_row*num_col, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        for (source=1; source<=numworkers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset][0], rows*num_col, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }
        t2 = MPI_Wtime();
        printf("Total time = %.2e seconds, here is the result matrix:\n", t2-t1);
        // for (i=0; i<num_row; i++) {
        //     for (j=0; j<num_col; j++)
        //         printf("%6.2f   ", matrix_c[i][j]);
        //     printf ("\n");
        // }
    }
    else{
        /* ------------- Worker -------------- */
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_a, rows*num_col, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_b, num_row*num_col, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        /* Matrix multiplication */
        for (k=0; k<num_col; k++) {
            for (i=0; i<rows; i++) {
                matrix_c[i][k] = 0.0;
                for (j=0; j<num_col; j++)
                matrix_c[i][k] = matrix_c[i][k] + matrix_a[i][j] * matrix_b[j][k];
            }
        }
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&matrix_c, rows*num_col, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

void print_matrix(double matrix[], int MAX_ROW,int MAX_COLUMN) {
    int i,j;
    for (i=0;i<MAX_ROW;i++){
        for (j=0;j<MAX_COLUMN;j++){
            printf("%lf ", matrix[ind_f(i,j, MAX_COLUMN)]);
        }
        printf("\n");
    }
}