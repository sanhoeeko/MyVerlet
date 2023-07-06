#define Real double

Real* icudaMalloc(int size);
void gpuupload(Real* c, Real* g, int size);
void gpudownload(Real* c, Real* g, int size);
void gpucpy(Real* g1, Real* g2, int size);

void cudaAdd(Real* a, Real* b, Real* res, int size);
void cudaMul(Real a, Real* vec, Real* res, int size);
void cudaPeriod(Real* x, int size);
void cudaReflect(Real* x, Real* vx, int size);
void cudaLj(Real* ax, Real* ay, Real* x, Real* y, int size);