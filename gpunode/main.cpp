#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include "RTSample.cuh"
#include "RTServer.h"

int main(int argc, char** argv)
{
	srand(time(NULL));

	//RunSample();

	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " tcp_host:port udp_host:port" << std::endl;
		return EXIT_FAILURE;
	}

	int err = 0;
	err = RunServers(argv[1], argv[2]);

	//i wszystko :D
	return err;
}


