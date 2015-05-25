#include <iostream>

using namespace CORE_API;
using namespace IR;
using namespace std;

int main(int argc, char** argv)
{
	auto def = CubeDef{};
	def.dims.emplace_back(5);
	def.dims.emplace_back(10);
	def.meas.emplace_back(Mea::Int);

	auto cube = RTCube{def};

	auto rows = Rows{2, 1, 1};
	rows[0].dims()[0] = 3;
	rows[0].dims()[1] = 7;
	rows[0].meas()[0].i = 55;

	cube.insert(rows);

	cube.query(Query{});

	cout << "OK" << endl;
}
