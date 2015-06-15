#include <iostream>
#include <sstream>
#include <fstream>

#include "../ir/core.h"
#include "../cubesql/parser.h"
#include "../to_ir/to_ir.h"
#include "RTServer.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	if (argc < 5)
	{
		std::cerr << "Usage: " << argv[0] << " core_type(dummy/cuda) cubedef_file tcp_host:port udp_host:port" << std::endl;
		return EXIT_FAILURE;
	}

	std::ifstream t(argv[2]);
	t.exceptions(std::istream::failbit | std::istream::badbit);
	std::stringstream buffer;
	buffer << t.rdbuf();

	auto sql = buffer.str();

	auto core = IR::Core{argv[1]};

	auto cubeSQLDef = CubeSQL::parseCubeDef(sql);
	auto cube = core.make_db(toIR(cubeSQLDef));

	int err = RunServers(cube, cubeSQLDef, argv[3], argv[4]);

	//i wszystko :D
	return err;
}


