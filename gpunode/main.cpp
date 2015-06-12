#include <iostream>
#include <sstream>
#include <fstream>

#include "../cudacore/api.h"
#include "../cubesql/parser.h"
#include "../to_ir/to_ir.h"
#include "RTServer.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " tcp_host:port udp_host:port" << std::endl;
		return EXIT_FAILURE;
	}

	std::ifstream t("gpunode/cube");
	t.exceptions(std::istream::failbit | std::istream::badbit);
	std::stringstream buffer;
	buffer << t.rdbuf();

	auto sql = buffer.str();

	auto cubeSQLDef = CubeSQL::parseCubeDef(sql);
	auto cube = CudaCore::RTCube(toIR(cubeSQLDef));

	int err = RunServers(cube, cubeSQLDef, argv[1], argv[2]);

	//i wszystko :D
	return err;
}


