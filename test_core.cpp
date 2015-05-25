#include <iostream>

using namespace CORE_API;
using namespace IR;
using namespace std;

int main(int argc, char** argv)
{
	RTCube cube{CubeDef{}};

	cube.insert(Rows{0, 0, 0});

	cube.query(Query{});

	cout << "OK" << endl;
}
