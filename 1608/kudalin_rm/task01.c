#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


int* GenerateMatrix(int rows_num, int columns_num) {
	int* matrix_as_vector = (int*)malloc(rows_num * columns_num * sizeof(int));
	for (int i = 0; i < rows_num * columns_num; ++i)
		matrix_as_vector[i] = rand() % 10;

	return matrix_as_vector;
}

void ShowMatrix(int* matrix, int rows_num, int columns_num) {
	printf("Generated matrix with size %d x %d:\n", rows_num, columns_num);
	int vector_index = 0;
	for (int i = 0; i < rows_num; ++i) {
		for (int j = 0; j < columns_num; ++j)
			printf("%d  ", matrix[vector_index++]);
		printf("\n");
	}
}


int ComputePartialSum(int* buf, int computation_volume) {
	int sum = 0;
	for (int i = 0; i < computation_volume; ++i)
		sum += buf[i];

	return sum;
}

void ExecSequentially(int* matrix, int size) {
	long sum = 0L;
	clock_t begin = clock();
	for (int i = 0; i < size; ++i)
			sum += matrix[i];
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
	int* matrix_as_vector = NULL;  // Vector representation
	int computation_volume = 0;  // Volume of computation per process
	int partial_sum = 0;         // Partial sum calculated by certain process
	long total_sum = 0L;         // Total sum of elements
	double begin = .0;           // Start time of parallel algorithm
	double end = .0;             // End time of parallel algorithm

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if (proc_rank == root) {
		rows_num = atoi(argv[1]);
		columns_num = atoi(argv[2]);
		matrix_as_vector = GenerateMatrix(rows_num, columns_num);
		if (rows_num * columns_num <= 100)
			ShowMatrix(matrix_as_vector, rows_num, columns_num);

		// Execute sequential algorithm
		ExecSequentially(matrix_as_vector, rows_num * columns_num);

		computation_volume = rows_num * columns_num / proc_num;

		begin = MPI_Wtime();
	}

	// Execute parallel algorithm
	MPI_Bcast(&computation_volume, 1, MPI_INT, root, MPI_COMM_WORLD);

	// Buffer that will hold a subset of elements for each process
	int* buf = (int*)malloc(computation_volume * sizeof(int));

	MPI_Scatter(matrix_as_vector, computation_volume, MPI_INT, buf, computation_volume, MPI_INT, root, MPI_COMM_WORLD);
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