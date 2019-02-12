
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t RellenaMatriz(int *x, int *y, int *m, unsigned int sizeX, unsigned int sizeY);

void imprimeVector(int *v, int n) {
	printf("{");
	for (int i = 1; i <= n; i++) {
		printf("%d", *v);
		if (i != n) printf(", ");
		v++;
	}
	printf("}");
}

void imprimeMatriz(int *v, int m, int n) {
	int i, j;
	printf("\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%d\t", v[i*n+j]);
		}
		printf("\n");
	}
}

__global__ void rmKernel(int *x, int *y, int *m) {
	int idx = threadIdx.x;
	int idy = blockIdx.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	x[idx] = idx;
	y[idy] = idy;
	m[id] = idy*10 + idx;
}

int main()
{
    const int sizeX = 5;
	const int sizeY = 6;
    int x[sizeX] = { 0, 0, 0, 0, 0 };
    int y[sizeY] = { 0, 0, 0, 0, 0, 0 };

	int m[sizeY*sizeX] =	{ 0, 0, 0, 0, 0 
							, 0, 0, 0, 0, 0
							, 0, 0, 0, 0, 0
							, 0, 0, 0, 0, 0
							, 0, 0, 0, 0, 0
							, 0, 0, 0, 0, 0 };

    // Add vectors in parallel.
	cudaError_t cudaStatus = RellenaMatriz(x, y, m, sizeX, sizeY);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Fallo en RellenaMatriz");
        return 1;
    }

	imprimeVector(x, sizeX);
	imprimeVector(y, sizeY);
	imprimeMatriz(m, sizeY, sizeX);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t RellenaMatriz(int *x, int *y, int *m, unsigned int sizeX, unsigned int sizeY)
{
    int *dev_x = 0;
    int *dev_y = 0;
	int *dev_m = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_x, sizeX * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_y, sizeY * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_m, sizeY * sizeX * sizeof(int *));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	/*
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x, x, sizeX * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_y, y, sizeY * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/

    // Launch a kernel on the GPU with one thread for each element.
	rmKernel <<<sizeY, sizeX>>>(dev_x, dev_y, dev_m);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(x, dev_x, sizeX * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(y, dev_y, sizeY * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(m, dev_m, sizeY * sizeX * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
    cudaFree(dev_x);
    cudaFree(dev_y);
	cudaFree(dev_m);
    
    return cudaStatus;
}
