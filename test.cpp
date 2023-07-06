#include<iostream>
#include"algo.h"
#include"matplotlibcpp.h"
using namespace std;
namespace plt = matplotlibcpp;

void scatter(Vec2d& xy) {
	Real* x = xy.x.cptr;
	Real* y = xy.y.cptr;
	int size = xy.x.size;
 
	vector<Real> x_vec(x, x + size);
	vector<Real> y_vec(y, y + size);
	plt::clf();
	plt::xlim(0, 1);
	plt::ylim(0, 1);
	plt::plot(x_vec, y_vec, ".");
	plt::pause(1e-6);
}

int main() {
	auto u = Particles2d(1024, 0.005);
	u.init(); // correct
	u.firstStep();
	for (int t = 0; t < 1000; t++) {
		u.step();
		u.download();
		scatter(*u.rc);
	}
	u.download();
}