#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include "kernel.cuh"

#define getidx int idx=blockIdx.x+threadIdx.x*1024

Real* icudaMalloc(int size) {
	Real* ptr;
	cudaMalloc(&ptr, size * sizeof(Real));
	return ptr;
}
void gpuupload(Real* c, Real* g, int size) {
	cudaMemcpy(g, c, size * sizeof(Real), cudaMemcpyHostToDevice);
}
void gpudownload(Real* c, Real* g, int size) {
	cudaMemcpy(c, g, size * sizeof(Real), cudaMemcpyDeviceToHost);
}
void gpucpy(Real* g1, Real* g2, int size) {
	cudaMemcpy(g2, g1, size * sizeof(Real), cudaMemcpyDeviceToDevice);
}

struct Pair {
	int block, thread;
};
Pair blockThreadAlloc(int size) {
	if (size <= 1024) {
		return { 1,size };
	}
	else {
		int blocks = size / 1024 + 1;
		return { blocks,1024 };
	}
}

__global__ void global_add(Real* a, Real* b, Real* res) {
	getidx;
	res[idx] = a[idx] + b[idx];
}
__global__ void global_mul(Real a, Real* vec, Real* res) {
	getidx;
	res[idx] = a * vec[idx];
}

__global__ void global_period(Real* x) {
	getidx;
	if (x[idx] < 0) {
		x[idx] = x[idx] + 1;
	}
	if (x[idx] > 1) {
		x[idx] = x[idx] - 1;
	}
}

__global__ void global_reflect(Real* x, Real* vx) {
	getidx;
	if (x[idx] < 0) {
		x[idx] = -x[idx];
		vx[idx] = -vx[idx];
	}
	if (x[idx] > 1) {
		x[idx] = 2 - x[idx];
		vx[idx] = -vx[idx];
	}
}

__global__ void global_lj(Real* ax, Real* ay, Real* x, Real* y, int size) {
	// constants in LJ potential
	const Real a = 2e-2;
	const Real b = 1e-4;
	const Real max_r2 = 4;
	const Real min_r2 = 1e-4; // prevent too large force
	// calculation
	getidx;
	Real x0 = x[idx];
	Real y0 = y[idx];
	Real fx = 0;
	Real fy = 0;
	for (int i = 0; i < size; i++) {
		if (i == idx)continue;
		Real dx = x[i] - x0;
		Real dy = y[i] - y0;
		Real r2 = dx * dx + dy * dy;
		if (r2 > max_r2)continue;
		if (r2 < min_r2)r2 = min_r2;
		Real a2_over_r2 = a * a / r2;
		Real u = 6 * b / (a * a) * (-2 * pow(a2_over_r2, 7) + pow(a2_over_r2, 4)); // \vec{F}=u\vec{r}
		fx += u * dx;
		fy += u * dy;
	}
	ax[idx] = fx;
	ay[idx] = fy;
}

void cudaAdd(Real* a, Real* b, Real* res, int size) {
	Pair bt = blockThreadAlloc(size);
	global_add << <bt.block, bt.thread >> > (a, b, res);
}
void cudaMul(Real a, Real* vec, Real* res, int size) {
	Pair bt = blockThreadAlloc(size);
	global_mul << <bt.block, bt.thread >> > (a, vec, res);
}

void cudaPeriod(Real* x, int size) {
	Pair bt = blockThreadAlloc(size);
	global_period << <bt.block, bt.thread >> > (x);
}

void cudaReflect(Real* x, Real* vx, int size)
{
	Pair bt = blockThreadAlloc(size);
	global_reflect << <bt.block, bt.thread >> > (x, vx);
}

void cudaLj(Real* ax, Real* ay, Real* x, Real* y, int size) {
	Pair bt = blockThreadAlloc(size);
	global_lj << <bt.block, bt.thread >> > (ax, ay, x, y, size);
}