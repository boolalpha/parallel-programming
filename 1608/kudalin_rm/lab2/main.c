#include "my_scatter.h"
#include <time.h>
#include <stdio.h>

#define DEFAULT_ARRAY_SIZE 10 
#define Ty int
#define MPI_Ty MPI_INT
#define FORMAT "%d "


int CheckEquality(void* src, void* res, int size) {
	for (int i = 0; i < size; ++i) {
		if (((Ty*)src)[i] != ((Ty*)res)[i]) {
			return 0;
		}
	}
	return 1;
}

Ty* GenerateArray(int size) {
	Ty* arr = (Ty*)malloc(size * sizeof(Ty));
	for (int i = 0; i < size; ++i) {
		arr[i] = (Ty)rand();
	}
	return arr;
}

void PrintArray(void* arr, int size) {
	for (int i = 0; i < size; ++i) {
		printf(FORMAT, ((Ty*)arr)[i]);
	}
	printf("\n");
}


int main(int argc, char* argv[]) {
	srand(time(NULL));
	int proc_num, proc_rank;
	int root = 0;             // Root process rank 
	double begin = .0;        // Start time
	double end = .0;          // End time

	int size = 0;
	void* src = NULL;
	void* dst = NULL;
	void* res = NULL;

	if (argc != 3) {
		printf("Error: should be 2 arguments!");
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

	if (proc_rank == root) {
		printf("Root rank = %d\n", root);
		size = atoi(argv[1]);
		if (size < 0) {
			printf("Error: size should be non-negative. Default size = %d\n",
				DEFAULT_ARRAY_SIZE);

			size = DEFAULT_ARRAY_SIZE;
		}
		src = GenerateArray(size);
		if (size <= 20) {
			PrintArray(src, size);
		}
		res = (Ty*)malloc(size * sizeof(Ty));
	}

	MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
	size /= proc_num;
	dst = (Ty*)malloc(size * sizeof(Ty));

	MPI_Barrier(MPI_COMM_WORLD);
	begin = MPI_Wtime();
	My_Scatter(src, size, MPI_Ty, dst, size, MPI_Ty, root, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();

	MPI_Gather(dst, size, MPI_Ty, res, size, MPI_Ty, root, MPI_COMM_WORLD);

	if (size <= 10) {
		printf("Rank %d received: ", proc_rank);
		PrintArray(dst, size);
	}

	if (proc_rank == root) {
		printf("\nMy_Scatter time = %lf\n", end - begin);
		printf("\nSTATUS = %d", CheckEquality(src, res, size * proc_num));
		free(src);
		free(res);
	}

	free(dst);

	MPI_Finalize();

	return 0;
}
