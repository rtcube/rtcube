#include <iostream>
#include <vector>
#include <signal.h>
#include <atomic>

#include "generator.h"

using namespace std;

atomic_bool should_exit_flag = false;

void usage() {
    std::cerr << "USAGE: " << std::endl;
    std::cerr << "data-generator addresses_filename [generator_id]" << std::endl;
    exit(0);
}

void sigint_handler(int signum) {
	*should_exit_flag = true;
}

int main(int argc, const char* argv[])
{
    std::vector<Generator::socket_info*> sockets;
    int generator_id = 0;

    if (argc < 2) {
        usage();
    }
	
	signal(SIGINT, signalHandler);

    sockets = Generator::LoadAddressesFile(argv[1]);

    if (argc == 3) {
        try {
            generator_id = std::stoi(argv[2]);
        }
        catch(const std::invalid_argument& ia) {
            std::cerr << "Invalid number of threads option '" << argv[2] << "'" << std::endl;
        }
    }

    Generator::GenerateData(&should_exit_flag, sockets, generator_id);
	if (*should_exit_flag) {
		std::cerr << "Terminated by user" << std::endl;
	}

    return 0;
}
