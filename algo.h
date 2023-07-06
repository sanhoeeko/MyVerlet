#pragma once
#include"vec2d.h"
#include"pool.h"
#include<random>
using namespace std;

void LJ(Vec2d& a, Vec2d& r);

Real uniform() {
	return (Real)rand() / RAND_MAX;
}
Real randcenter() {
	return (Real)rand() / RAND_MAX - 0.5;
}

void boundary(Vec2d& r, Vec2d& v) {
	ireflect(r.x.gptr, v.x.gptr, v.x.size);
	ireflect(r.y.gptr, v.y.gptr, v.y.size);
}

/*
* Insight: do not store particles using class Particle, but regard them as vectors.
* (For simplicity, double buffer trick is omitted.)
*/
class Particles2d {
public:
	int n;
	Vec2d r;
	Vec2d v;
	Vec2d a;
	// cache
	Pool<Vec2d> cache;
	Vec2d* rc;
	Vec2d* vc;
	Vec2d* ac;
	// others
	bool* colored;
	double dt;

	Particles2d(int n, Real dt) {
		this->n = n;
		r = Vec2d(n);
		v = Vec2d(n);
		a = Vec2d(n);
		cache = Pool<Vec2d>(n);
		rc = poolapply(cache);
		vc = poolapply(cache);
		ac = poolapply(cache);
		cache.pin();
		colored = (bool*)malloc(sizeof(bool) * n);
		this->dt = dt;
	}
	void upload() {
		r.upload();
		v.upload();
		a.upload();
	}
	void download() {
		r.download();
		v.download();
		a.download();
	}
	void set(Vec2d*& p_cache, Vec2d p) {
		cache.set(p_cache, &p);
		p_cache = &p;
		p_cache->pool = &cache; // patch
	}
	void init() {
		for (int i = 0; i < n; i++) {
			r.x.cptr[i] = uniform();
			r.y.cptr[i] = uniform();
			v.x.cptr[i] = randcenter();
			v.y.cptr[i] = randcenter();
		}
		upload();
		firstStep();
	}
	void firstStep() {
		calForce();
		a.copyto(*ac);
	}
	void calForce() {
		// LJ potential for finite screen, zero boundary condition
		LJ(a, r);
	}
	void singleFrog() {
		set(rc, r);
		set(vc, v);
		set(ac, a);
		v = *vc + (*ac + a) * (dt / 2); //计算结果未成功获取(GPU)
		r = *rc + *vc * dt + *ac * (dt * dt / 2);
		boundary(r, v);
		cache.restore();
	}
	void step() {
		calForce();
		singleFrog();
	}
};

void cpu_single_lj(Real* ax, Real* ay, Real* x, Real* y, int size, int idx) {
	// constants in LJ potential
	const Real a = 2e-2;
	const Real b = 1e-4;
	const Real max_r2 = 4;
	const Real min_r2 = 1e-4; // prevent too large force
	// calculation
	// getidx;
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

void cpuLj(Real* ax, Real* ay, Real* x, Real* y, int size) {
	loop(size) {
		cpu_single_lj(ax, ay, x, y, size, i);
	}
}

void LJ(Vec2d& a, Vec2d& r) {
#if debug
	cpuLj(a.x.gptr, a.y.gptr, r.x.gptr, r.y.gptr, a.x.size);
#else
	cudaLj(a.x.gptr, a.y.gptr, r.x.gptr, r.y.gptr, a.x.size);
#endif
}