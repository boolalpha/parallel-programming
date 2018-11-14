#define MSMPI_NO_DEPRECATE_20

#include "my_scatter.h"

int My_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

	int proc_num, proc_rank;
	int error_code = MPI_SUCCESS;  // Error code
	MPI_Aint send_ext, recv_ext;   // Extents of sendtype and recvtype
	char* tmp_buf = NULL;          // Buffer containing part of elements for the current process
	int tree_order = 0;            // Order of the tree
	int origin_root = 0;           // Origin rank of the root process

	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Type_extent(sendtype, &send_ext);
	MPI_Type_extent(recvtype, &recv_ext);

	if (proc_rank == root && root != 0) {
		origin_root = root;
		MPI_Send(sendbuf, proc_num * sendcount, sendtype, 0, 0, MPI_COMM_WORLD);

	} else if (proc_rank == 0 && proc_rank != root) {
		origin_root = root;
		tmp_buf = (char*)malloc(proc_num * sendcount * send_ext);
		MPI_Recv(tmp_buf, proc_num * sendcount, sendtype, root, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	root = 0;

	if (proc_rank % 2) {
		error_code = MPI_Recv(recvbuf, recvcount, recvtype, proc_rank - 1, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else {
		if (proc_rank == root) {
			if (tmp_buf == NULL) {
				tmp_buf = (char*)sendbuf;
			}
			tree_order = GetTreeOrder(proc_num);

			for (int i = 0; i < tree_order; ++i) {
				int recv_rank = 1 << i;
				int recv_buf_size = GetBufferSize(recv_rank, proc_num) * recvcount;
				error_code = MPI_Send(tmp_buf + recv_rank * sendcount * send_ext,
					recv_buf_size, sendtype, recv_rank, 0, MPI_COMM_WORLD);
			}

		} else {
			int buf_size = GetBufferSize(proc_rank, proc_num) * sendcount;
			tmp_buf = (char*)malloc(buf_size * send_ext);
			error_code = MPI_Recv(tmp_buf, buf_size, recvtype, MPI_ANY_SOURCE, 0,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			int i = 1;
			while (!(proc_rank & i) && ((proc_rank | i) < proc_num)) {
				int recv_rank = proc_rank | i;
				int recv_buf_size = GetBufferSize(recv_rank, proc_num) * recvcount;
				error_code = MPI_Send(tmp_buf + i * sendcount * send_ext, recv_buf_size,
					sendtype, recv_rank, 0, MPI_COMM_WORLD);
				i <<= 1;
			}
		}
		for (int i = 0; i < recvcount * recv_ext; ++i) {
			((char*)recvbuf)[i] = tmp_buf[i];
		}
	}

	// Cleaning up
	if (proc_rank != root || (proc_rank == root && origin_root != 0)) {
		free(tmp_buf);
	}

	return error_code;
}

int GetBufferSize(int proc_rank, int proc_num) {
	int i = 1;
	int size = 1;
	while (!(proc_rank & i) && ((proc_rank | i) < proc_num)) {
		size *= 2;
		i <<= 1;
	}
	return size;
}

int GetTreeOrder(int proc_num) {
	int x = proc_num - 1;
	int order = 0;
	while (x) {
		order++;
		x >>= 1;
	}
	return order;
}
