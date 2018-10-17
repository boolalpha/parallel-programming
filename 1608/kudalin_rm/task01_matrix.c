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
    MPI_Status status;
    const int root = 0;          // Root process rank

    T** matrix = NULL;
    int rows_num = 0;            
    int columns_num = 0;
    int computation_volume = 0;  // Volume of computation per process (number of rows)
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
        matrix = GenerateMatrix(rows_num, columns_num);
        ShowMatrix(matrix, rows_num, columns_num);

        // Execute sequential algorithm
        ExecSequentially(matrix, rows_num, columns_num);

        computation_volume = rows_num / proc_num;
        begin = MPI_Wtime();
    }
    
    MPI_Bcast(&computation_volume, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&columns_num, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (proc_rank == root) {
        int row_index = computation_volume;  // row index for the next process

        for (int i = 1; i < proc_num; ++i)
            for (int j = 0; j < computation_volume; ++j)
                MPI_Send(matrix[row_index++], columns_num, MPI_T, i, 0, MPI_COMM_WORLD);

        // Root process part
        for (int i = 0; i < computation_volume; ++i)
            for (int j = 0; j < columns_num; ++j)
                partial_sum += matrix[i][j];
    }
    else {
        T* buf = (T*)malloc(columns_num * sizeof(T));
        for (int i = 0; i < computation_volume; ++i) {
            MPI_Recv(buf, columns_num, MPI_T, root, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < columns_num; ++i)
                partial_sum += buf[i];
        }
        free(buf);
    }

    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    if (proc_rank == root) {
        // Remaining elements, if any, are summarized by the root process
        for (int i = computation_volume * proc_num; i < rows_num; ++i)
            for (int j = 0; j < columns_num; ++j)
                total_sum += matrix[i][j];
        end = MPI_Wtime();
        printf("Parallel algorithm: sum = %li, time spent = %f\n", total_sum, end - begin);
        
        DeleteMatrix(matrix, rows_num);
    }

    MPI_Finalize();
    return 0;
}