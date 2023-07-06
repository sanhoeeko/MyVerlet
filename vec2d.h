#pragma once
#include"cudaBridge.h"
#include"pool.h"

class Vec2d; // forward declaration (¿‡«∞÷√…˘√˜)

Vec2d* poolapply(Pool<Vec2d>& p);

/*
* analog class GpuMemory 
*/
class Vec2d {
public:
	GpuMemory x;
	GpuMemory y;
	Pool<Vec2d>* pool;

	Vec2d() {

	}
	Vec2d(int n) {
		x = GpuMemory(n);
		y = GpuMemory(n);
		pool = NULL;
	}
	Vec2d(Vec2d& v) {
		x = v.x;
		y = v.y;
		pool = v.pool;
	}
	Vec2d(GpuMemory& x, GpuMemory& y, Pool<Vec2d>* pool) {
		this->x = x;
		this->y = y;
		this->pool = pool;
	}
	void upload() {
		x.upload();
		y.upload();
	}
	void download() {
		x.download();
		y.download();
	}
	void copyto(Vec2d& v) {
		x.copyto(v.x);
		y.copyto(v.y);
	}
	// pack inplace operator to overload operator
	Vec2d operator+(Vec2d& v) {
		Pool<Vec2d>* p = pool;
		Vec2d* res = poolapply(*p);
		res->x.to_add(x, v.x);
		res->y.to_add(y, v.y);
		return *res; // memory leak but will be cleared each iteration
	}
	Vec2d operator*(Real a) {
		Vec2d* res = pool->apply();
		res->x.to_mul(a, x);
		res->y.to_mul(a, y);
		return *res;
	}
};

Vec2d* poolapply(Pool<Vec2d>& p) {
	auto res = p.apply();
	res->pool = &p;
	return res;
}