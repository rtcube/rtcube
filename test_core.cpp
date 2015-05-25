#include <iostream>

using namespace CORE_API;
using namespace IR;
using namespace std;

int main(int argc, char** argv)
{
	auto cube = RTCube{CubeDef{}};

	cube.insert(Rows{0, 0, 0});

	cube.query(Query{});

	cout << "OK" << endl;
}
