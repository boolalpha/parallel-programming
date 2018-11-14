#define MSMPI_NO_DEPRECATE_20
#define DEFAULT_ARRAY_SIZE 10 

#include "my_scatter.h"
#include <time.h>
#include <stdio.h>
#include <string.h>


enum ElemType { INTEGER, FLOAT, DOUBLE };

int* GenerateArrayOfIntegers(int size) {
	int* arr = (int*)malloc(size * sizeof(int));
	for (int i = 0; i < size; ++i) {
		arr[i] = rand() % 10;
	}

	return arr;
}

float* GenerateArrayOfFloats(int size) {
	float* arr = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; ++i) {
		arr[i] = (float)(rand() / (double)RAND_MAX);
	}

	return arr;
}

double* GenerateArrayOfDoubles(int size) {
	double* arr = (double*)malloc(size * sizeof(double));
	for (int i = 0; i < size; ++i) {
		arr[i] = rand() / (double)RAND_MAX;
	}

	return arr;
}

void PrintArray(void* arr, enum ElemType type, int size) {
	for (int i = 0; i < size; ++i) {
		switch (type) {
		case FLOAT:
			printf("%.3f ", ((float*)arr)[i]);
			break;
		case DOUBLE:
			printf("%.3f ", ((double*)arr)[i]);
			break;
		case INTEGER: default:
			printf("%d ", ((int*)arr)[i]);
		}
	}
	printf("\n");
}


int main(int argc, char* argv[]) {
	srand(time(NULL));
	int proc_num, proc_rank;
	int root = 0;                  // Root process rank 
	double begin = .0;             // Start time
	double end = .0;               // End time
	enum ElemType type = INTEGER;  // Type of elements

	int size = 0;
	void* arr = NULL;

	if (argc != 4) {
		printf("Error: should be 3 arguments!");
		exit(EXIT_FAILURE);
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	root = atoi(argv[2]);
	if (root < 0 || root >= proc_num) {
		printf("Error: wrong root process rank!");
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if (strcmp(argv[3], "-i") == 0) {
		type = INTEGER;
	} else if (strcmp(argv[3], "-f") == 0) {
		type = FLOAT;
	} else if (strcmp(argv[3], "-d") == 0) {
		type = DOUBLE;
	}

	if (proc_rank == root) {
		printf("Root rank = %d\n", root);
		size = atoi(argv[1]);
		if (size < 0) {
			printf("Error: size should be non-negative. Default size = %d\n",
				DEFAULT_ARRAY_SIZE);

			size = DEFAULT_ARRAY_SIZE;
		}
		switch (type) {
		case FLOAT:
			arr = GenerateArrayOfFloats(size);
			break;
		case DOUBLE:
			arr = GenerateArrayOfDoubles(size);
			break;
		case INTEGER: default:
			arr = GenerateArrayOfIntegers(size);
		}
		if (size <= 20) {
			PrintArray(arr, type, size);
		}
		printf("\n");
	}

	MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
	size /= proc_num;

	void* buf = NULL;
	switch (type) {
	case FLOAT:
		buf = malloc(size * sizeof(float));
		MPI_Barrier(MPI_COMM_WORLD);
		begin = MPI_Wtime();
		My_Scatter(arr, size, MPI_FLOAT, buf, size, MPI_FLOAT, root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		end = MPI_Wtime();
		break;
	case DOUBLE:
		buf = malloc(size * sizeof(double));
		MPI_Barrier(MPI_COMM_WORLD);
		begin = MPI_Wtime();
		My_Scatter(arr, size, MPI_DOUBLE, buf, size, MPI_DOUBLE, root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		end = MPI_Wtime();
		break;
	case INTEGER: default:
		buf = malloc(size * sizeof(int));
		MPI_Barrier(MPI_COMM_WORLD);
		begin = MPI_Wtime();
		My_Scatter(arr, size, MPI_INT, buf, size, MPI_INT, root, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		end = MPI_Wtime();
	}

	if (size <= 10) {
		printf("Rank %d received: ", proc_rank);
		PrintArray(buf, type, size);
	}

	if (proc_rank == root) {
		printf("\nMy_Scatter time = %lf\n", end - begin);
		free(arr);
	}

	free(buf);

	MPI_Finalize();
	return 0;
}