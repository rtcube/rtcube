#include <iostream>

#include "RTCubeApi.h"
#include "RTServer.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " tcp_host:port udp_host:port" << std::endl;
		return EXIT_FAILURE;
	}

	initCube();

	int err = RunServers(argv[1], argv[2]);

	//i wszystko :D
	return err;
}


