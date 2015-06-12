#include <iostream>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdexcept>

#include "Generator.cpp"

using namespace std;

void usage() {
    std::cerr << "USAGE: " << std::endl;
    std::cerr << "row-generator addresses_filename cube_filename [no_threads]" << std::endl;
    exit(0);
}

int main(int argc, const char* argv[]) {
    Generator::cube_info * cube;
    std::vector<Generator::socket_info*> sockets;
    int no_blocks, no_threads = 2;
    std::string addresses_filepath, cube_filepath;

    if (argc < 3) {
        usage();
    }

    addresses_filepath = argv[1];
    sockets = Generator::LoadAddressesFile(addresses_filepath);

    cube_filepath = argv[2];
    if (!(cube = Generator::LoadCubeFile(cube_filepath))) {
        std::cerr << "Couldn't load cube file " << cube_filepath << std::endl;
        return 1;
    }
    if (argc == 4) {
        try {
            no_threads = std::stoi(argv[3]);
        }
        catch(const std::invalid_argument& ia) {
            std::cerr << "Invalid number of threads option '" << argv[3] << "'" << std::endl;
        }
    }

    std::string line;

    while(true) {
        // main loop

        std::cout << "%d - generate %d blocks; Q|q - quit:" << std::endl;
        std::getline(std::cin, line);

        // Q|q - exit
        if(line[0] == 'q' || line[0] == 'Q') {
            break;
        }

        // %d - Generate rows
        try {
            no_blocks = std::stoi(line);
            Generator::StartGenerating(no_blocks,cube, sockets, no_threads);
        }
        catch (const std::invalid_argument& ia) {
            //std::cerr << "Invalid argument: " << ia.what() << '\n';
            std::cerr << "Unknown option '" << line << "'" << std::endl;
            continue;
        }
    }

    return 0;
}
