#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define KERNEL_SIZE 3
#define PI_NUM 3.14159265358979323846
#define VIEW_THRESHOLD 20
#define DEFAULT_IMAGE_SIZE 10000


unsigned char* LoadImage(int height, int width) {
	unsigned char* image =
		(unsigned char*)malloc(height * width * sizeof(unsigned char));

	for (int i = 0; i < height * width; ++i) {
		image[i] = rand() % 256;
	}

	return image;
}

void ShowImage(unsigned char* image, int height, int width) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			printf("%hhu ", image[j + i * width]);
		}
		printf("\n");
	}
}

void ShowKernel(double* kernel) {
	for (int i = 0; i < KERNEL_SIZE; ++i) {
		for (int j = 0; j < KERNEL_SIZE; ++j) {
			printf("%.5f ", kernel[j + i * KERNEL_SIZE]);
		}
		printf("\n");
	}
}

double* CreateGaussianKernel(double sigma) {
	double* kernel =
		(double*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double));

	double coeff = 1 / (sigma * sigma * 2 * PI_NUM);
	int radius = KERNEL_SIZE / 2;
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			kernel[j + radius + (i + radius) * KERNEL_SIZE] =
				coeff * exp(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}

	return kernel;
}

unsigned char* ProcessImage(unsigned char* image, double* kernel,
	int height, int width, int start, int end) {

	unsigned char* result_image =
		(unsigned char*)malloc((end - start) * width * sizeof(unsigned char));

	int radius = KERNEL_SIZE / 2;
	for (int i = start; i < end; ++i) {
		for (int j = 0; j < width; ++j) {
			double result_color = .0;
			unsigned char adjacent_color = 0;
			for (int k = -radius; k <= radius; ++k) {
				for (int q = -radius; q <= radius; ++q) {
					if ((j + q) >= 0 && (j + q) < width &&
						(i + k) >= 0 && (i + k) < height) {

						adjacent_color = image[j + q + (i + k) * width];

						result_color += adjacent_color * kernel[q + radius
							+ (k + radius) * KERNEL_SIZE];
					}
				}
			}
			result_image[j + (i - start) * width] =
				(unsigned char)((result_color <= 255) ? result_color : 255);
		}
	}

	return result_image;
}

int CheckEquality(unsigned char* seq_res_img, unsigned char* par_res_img,
	int height, int width) {

	for (int i = 0; i < height * width; ++i) {
		if (seq_res_img[i] != par_res_img[i]) {
			return 0;
		}
	}

	return 1;
}

int main(int argc, char** argv) {
	srand(time(NULL));

	int proc_num, proc_rank;
	const int root = 0;
	double par_begin, par_end;
	int* counts = NULL;
	int* displs = NULL;

	double seq_begin, seq_end;

	int height, width, original_height;
	int extended_img_part_height;
	int remaining_rows;
	int sent_rows_num;
	int start_row_index, end_row_index;
	int offset;

	unsigned char* image = NULL;
	unsigned char* seq_res_img = NULL;
	unsigned char* par_res_img = NULL;
	unsigned char* proc_img_part = NULL;
	unsigned char* proc_res_img = NULL;
	double* kernel = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if (root >= proc_num || proc_num == 1) {
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if (proc_rank == root) {
		if (argc != 3) {
			height = width = DEFAULT_IMAGE_SIZE;
		} else {
			height = atoi(argv[1]);
			width = atoi(argv[2]);
		}
		if (height <= 0) {
			height = DEFAULT_IMAGE_SIZE;
		}
		if (width <= 0) {
			width = DEFAULT_IMAGE_SIZE;
		}

		original_height = height;

		image = LoadImage(height, width);

		if (height <= VIEW_THRESHOLD) {
			printf("\nOriginal image:\n");
			ShowImage(image, height, width);
		}

		kernel = CreateGaussianKernel(1);

		// Execute sequential algorithm
		seq_begin = MPI_Wtime();
		seq_res_img = ProcessImage(image, kernel, height, width, 0, height);
		seq_end = MPI_Wtime();

		if (height <= VIEW_THRESHOLD) {
			printf("\nAfter sequential processing:\n");
			ShowImage(seq_res_img, height, width);
		}

		printf("\nSequential algorithm took: %f\n", seq_end - seq_begin);
		
		par_res_img =
			(unsigned char*)malloc(height * width * sizeof(unsigned char));

	} else {
		kernel = (double*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(double));
	}

	// Execute parallel algorithm
	MPI_Barrier(MPI_COMM_WORLD);
	par_begin = MPI_Wtime();

	MPI_Bcast(&height, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&width, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(kernel, KERNEL_SIZE * KERNEL_SIZE, MPI_DOUBLE,
		root, MPI_COMM_WORLD);

	if (proc_num > height) {
		printf("proc_num should be <= height");
		free(image);
		free(seq_res_img);
		free(par_res_img);
		free(kernel);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	remaining_rows = height % proc_num;

	height /= proc_num;

	counts = (int*)malloc(proc_num * sizeof(int));
	displs = (int*)malloc(proc_num * sizeof(int));

	sent_rows_num = 0;
	for (int i = 0; i < proc_num; ++i) {
		counts[i] = height * width;
		if (remaining_rows > 0) {
			counts[i] += width;
			remaining_rows--;
		}

		displs[i] = sent_rows_num;
		sent_rows_num += counts[i];
	}

	offset = width;
	start_row_index = 1;
	extended_img_part_height = counts[proc_rank] / width + 1;

	if (proc_rank == 0) {
		offset = 0;
		start_row_index = 0;
		end_row_index = extended_img_part_height - 1;
		proc_img_part =
			(unsigned char*)malloc((counts[proc_rank] + width) *
				sizeof(unsigned char));

	} else if (proc_rank == proc_num - 1) {
		end_row_index = extended_img_part_height;
		proc_img_part =
			(unsigned char*)malloc((counts[proc_rank] + width) *
				sizeof(unsigned char));

	} else {
		extended_img_part_height++;
		end_row_index = extended_img_part_height - 1;
		proc_img_part =
			(unsigned char*)malloc((counts[proc_rank] + 2 * width)
				* sizeof(unsigned char));
	}

	MPI_Scatterv(image, counts, displs, MPI_UNSIGNED_CHAR, proc_img_part + offset,
		counts[proc_rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

	if (proc_rank != 0) {
		MPI_Sendrecv(proc_img_part + offset, width, MPI_UNSIGNED_CHAR,
			proc_rank - 1, 0, proc_img_part, width, MPI_UNSIGNED_CHAR,
			proc_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	if (proc_rank != proc_num - 1) {
		MPI_Sendrecv(proc_img_part + offset + counts[proc_rank] - width, width,
			MPI_UNSIGNED_CHAR, proc_rank + 1, 0,
			proc_img_part + offset + counts[proc_rank], width, MPI_UNSIGNED_CHAR,
			proc_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	proc_res_img = ProcessImage(proc_img_part, kernel,
		extended_img_part_height, width, start_row_index, end_row_index);

	MPI_Gatherv(proc_res_img, counts[proc_rank], MPI_UNSIGNED_CHAR,
		par_res_img, counts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	par_end = MPI_Wtime();

	if (proc_rank == root) {
		if (original_height <= VIEW_THRESHOLD) {
			printf("\nAfter parallel processing:\n");
			ShowImage(par_res_img, original_height, width);
		}
		printf("STATUS = %d, parallel algorithm took: %f\n",
			CheckEquality(seq_res_img, par_res_img, original_height, width),
			par_end - par_begin);

		free(image);
		free(seq_res_img);
		free(par_res_img);
	}

	free(counts);
	free(displs);
	free(proc_img_part);
	free(proc_res_img);
	free(kernel);

	MPI_Finalize();

	return 0;
}