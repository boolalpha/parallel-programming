#ifndef MY_SCATTER_H
#define MY_SCATTER_H

#include <mpi.h>
#include <stdlib.h>

int My_Scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

int GetBufferSize(int proc_rank, int proc_num);
int GetTreeOrder(int proc_num);

#endif