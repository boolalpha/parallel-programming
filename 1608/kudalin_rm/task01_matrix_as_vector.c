#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef int T;
#define MPI_T MPI_INT


T** GenerateMatrix(int rows_num, int columns_num) {
    T** matrix = (T**)malloc(rows_num * sizeof(T*));
    for (int i = 0; i < rows_num; ++i)
        matrix[i] = (T*)malloc(columns_num * sizeof(T));

    for (int i = 0; i < rows_num; ++i)
        for (int j = 0; j < columns_num; ++j)
            matrix[i][j] = rand() % 10;

    return matrix;
}

void ShowMatrix(T** matrix, int rows_num, int columns_num) {
    printf("Generated matrix with size %d x %d:\n", rows_num, columns_num);
    for (int i = 0; i < rows_num; ++i) {
        for (int j = 0; j < columns_num; ++j)
            printf("%d  ", matrix[i][j]);
        printf("\n");
    }
}

void DeleteMatrix(T** matrix, int rows_num) {
    for (int i = 0; i < rows_num; ++i)
        free(matrix[i]);
    free(matrix);
}

T* GetVectorRepresentation(T** matrix, int rows_num, int columns_num) {
    T* matrix_as_vector = (T*)malloc(rows_num * columns_num * sizeof(T));
    int vector_index = 0;
    for (int i = 0; i < rows_num; ++i)
        for (int j = 0; j < columns_num; ++j)
            matrix_as_vector[vector_index++] = matrix[i][j];

    return matrix_as_vector;
}

int ComputePartialSum(T* buf, int computation_volume) {
    int sum = 0;
    for (int i = 0; i < computation_volume; ++i)
        sum += buf[i];

    return sum;
}

void ExecSequentially(T** matrix, int rows_num, int columns_num) {
    long sum = 0L;
    clock_t begin = clock();
    for (int i = 0; i < rows_num; ++i)
        for (int j = 0; j < columns_num; ++j)
            sum += matrix[i][j];
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Sequential algorithm: sum = %li, time spent = %f\n", sum, time_spent);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int proc_num, proc_rank;
    const int root = 0;          // Root process rank

    int rows_num = 0;
    int columns_num = 0;
    T* matrix_as_vector = NULL;  // Vector representation
    int computation_volume = 0;  // Volume of computation per process
    int partial_sum = 0;         // Partial sum calculated by certain process
    long total_sum = 0L;         // Total sum of elements
    double begin = .0;           // Start time of parallel algorithm
    double end = .0;             // End time of parallel algorithm

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if (proc_rank == root) {
        rows_num = 10;
        columns_num = 10;
        T** matrix = GenerateMatrix(rows_num, columns_num);
        ShowMatrix(matrix, rows_num, columns_num);

        // Execute sequential algorithm
        ExecSequentially(matrix, rows_num, columns_num);

        computation_volume = rows_num * columns_num / proc_num;
        matrix_as_vector = GetVectorRepresentation(matrix, rows_num, columns_num);
        DeleteMatrix(matrix, rows_num);

        begin = MPI_Wtime();
    }

    // Execute parallel algorithm
    MPI_Bcast(&computation_volume, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Buffer that will hold a subset of elements for each process
    T* buf = (T*)malloc(computation_volume * sizeof(T));

    MPI_Scatter(matrix_as_vector, computation_volume, MPI_T, buf, computation_volume, MPI_T, root, MPI_COMM_WORLD);
    partial_sum = ComputePartialSum(buf, computation_volume);

    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    if (proc_rank == root) {
        // Remaining elemetns, if any, are summarized by the root process
        for (int i = computation_volume * proc_num; i < rows_num * columns_num; ++i)
            total_sum += matrix_as_vector[i];
        end = MPI_Wtime();
        printf("Parallel algorithm: sum = %li, time spent = %f\n", total_sum, end - begin);

        free(matrix_as_vector);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}