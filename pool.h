#pragma once

using namespace std;

#define POOL_MAX 16

template<typename t>
class Pool {
public:
	t* pool[POOL_MAX];
	bool occupied[POOL_MAX];
	bool pinned[POOL_MAX];

	Pool() {

	}
	Pool(int n) { // n is the parameter of the construct function of class t
		//allocate memory space
		for (int i = 0; i < POOL_MAX; i++) {
			pool[i] = new t(n);
			occupied[i] = false;
			pinned[i] = false;
		}
	}
	t* apply() {
		for (int i = 0; i < POOL_MAX; i++) {
			if (!occupied[i]) {
				occupied[i] = true;
				return pool[i];
			}
		}
		throw "Pool full!";
	}
	void kill(t* p) {
		for(int i = 0;i < POOL_MAX;i++) {
			if (pool[i] == p) {
				occupied[i] = false;
				break;
			}
		}
	}
	void pin() {
		for (int i = 0; i < POOL_MAX; i++) {
			pinned[i] = occupied[i];
		}
	}
	void restore() {
		for (int i = 0; i < POOL_MAX; i++) {
			if (!pinned[i]) {
				occupied[i] = false; // delete objects that are not pinned
			}
		}
	}
	void set(t* p_cache, t* p) {
		for (int i = 0; i < POOL_MAX; i++) {
			if (pool[i] == p_cache) {
				pool[i] = p; break;
			}
		}
	}
	t& operator[](int id) {
		return *(pool[id]);
	}
};