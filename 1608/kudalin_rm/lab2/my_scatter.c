#define MSMPI_NO_DEPRECATE_20

#include "my_scatter.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int My_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

	int proc_num, proc_rank;
	int error_code = MPI_SUCCESS;  // Error code
	MPI_Aint send_ext, recv_ext;   // Extents of sendtype and recvtype
	char* tmp_buf = NULL;          // Buffer containing part of elements for the current process
	int buf_size;				   // Buffer size
	int recv_rank, recv_buf_size;
	int i;

	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Type_extent(sendtype, &send_ext);
	MPI_Type_extent(recvtype, &recv_ext);

	proc_rank -= root;
	if (proc_rank < 0) {
		proc_rank += proc_num;
	}

	buf_size = GetNodesNumber(proc_rank, proc_num) * recvcount;
	tmp_buf = (char*)malloc(buf_size * recv_ext);

	if (proc_rank == 0) {
		memcpy(tmp_buf, (char*)sendbuf + root * sendcount * send_ext, (buf_size - root * sendcount) * send_ext);
		memcpy(tmp_buf + (buf_size - root * sendcount) * send_ext, sendbuf, root * sendcount * send_ext);
	} else {
		error_code = MPI_Recv(tmp_buf, buf_size, recvtype, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	i = (1 << GetSubtreeOrder(proc_rank, proc_num)) >> 1;
	while (i > 0) {
		if ((proc_rank | i) < proc_num) {
			recv_rank = proc_rank | i;
			recv_buf_size = GetNodesNumber(recv_rank, proc_num) * recvcount;
			error_code = MPI_Send(tmp_buf + i * sendcount * send_ext, recv_buf_size,
				sendtype, (recv_rank + root) % proc_num, 0, MPI_COMM_WORLD);
		}
		i >>= 1;
	}

	memcpy(recvbuf, tmp_buf, recvcount * recv_ext);

	free(tmp_buf);

	return error_code;
}

int GetNodesNumber(int proc_rank, int proc_num) {
	int size = 1 << GetSubtreeOrder(proc_rank, proc_num);
	return (proc_num - proc_rank < size) ? proc_num - proc_rank : size;
}

int GetSubtreeOrder(int proc_rank, int proc_num) {
	int i = 1;
	int order = 0;
	if (proc_rank == 0) {
		return (int)ceil(log2(proc_num));
	} else {
		while (!(proc_rank & i)) {
			++order;
			i <<= 1;
		}
		return order;
	}
}
