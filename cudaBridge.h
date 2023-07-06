#pragma once
#include"kernel.cuh"
#include<stdlib.h>
#include<string.h>

#define debug false
#define loop(__n) for(int i=0;i<__n;i++)

#if debug
#define imalloc cpumalloc
#define iupload cpuupload
#define idownload cpudownload
#define icopy cpucpy
#define iadd cpuadd
#define imul cpumul
#define iperiod cpuperiod
#define ireflect cpureflect
#else
#define imalloc gpumalloc
#define iupload gpuupload
#define idownload gpudownload
#define icopy gpucpy
#define iadd gpuadd
#define imul gpumul
#define iperiod gpuperiod
#define ireflect gpureflect
#endif

Real* cpumalloc(int size) {
	return (double*)malloc(sizeof(Real) * size);
}
Real* gpumalloc(int size) {
	return icudaMalloc(size);
}

void cpuupload(Real* c, Real* g, int size) {
	memcpy(g, c, size * sizeof(Real));
}
// gpuupload: see kernel.cu
void cpudownload(Real* c, Real* g, int size) {
	memcpy(c, g, size * sizeof(Real));
}
// gpudownload: see kernel.cu
void cpucpy(Real* c1, Real* c2, int size) {
	memcpy(c2, c1, size * sizeof(Real));
}
// gpucpy: see kernel.cu

void cpuadd(Real* a, Real* b, Real* res, int size) {
	loop(size) {
		res[i] = a[i] + b[i];
	}
}
void gpuadd(Real* a, Real* b, Real* res, int size) {
	cudaAdd(a, b, res, size);
}
void cpumul(Real a, Real* vec, Real* res, int size) {
	loop(size) {
		res[i] = a * vec[i];
	}
}
void gpumul(Real a, Real* vec, Real* res, int size) {
	cudaMul(a, vec, res, size);
}

void cpuperiod(Real* x, int size) {
	loop(size) {
		if (x[i] < 0) {
			x[i] = x[i] + 1;
		}
		if (x[i] > 1) {
			x[i] = x[i] - 1;
		}
	}
}
void gpuperiod(Real* x, int size) {
	cudaPeriod(x, size);
}

void cpureflect(Real* x, Real* vx, int size) {
	loop(size) {
		if (x[i] < 0) {
			x[i] = -x[i];
			vx[i] = -vx[i];
		}
		if (x[i] > 1) {
			x[i] = 2 - x[i];
			vx[i] = -vx[i];
		}
	}
}
void gpureflect(Real* x, Real* vx, int size) {
	cudaReflect(x, vx, size);
}

class GpuMemory {
public:
	int size;
	Real* cptr;
	Real* gptr;

	GpuMemory() {

	}
	GpuMemory(int size) {
		this->size = size;
		cptr = cpumalloc(size);
		gptr = imalloc(size);
	}
	GpuMemory(GpuMemory& x) {
		size = x.size;
		cptr = x.cptr;
		gptr = x.gptr;
	}
	void upload() {
		iupload(cptr, gptr, size);
	}
	void download() {
		idownload(cptr, gptr, size);
	}
	void copyto(GpuMemory& x) {
		icopy(gptr, x.gptr, size);
	}
	// inplace operator
	void to_add(GpuMemory& x, GpuMemory& y) {
		iadd(x.gptr, y.gptr, gptr, size);
	}
	void to_mul(Real a, GpuMemory& x) {
		imul(a, x.gptr, gptr, size);
	}
};