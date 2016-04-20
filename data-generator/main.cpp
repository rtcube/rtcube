#include <iostream>
#include <vector>

#include "generator.h"

using namespace std;

void usage() {
    std::cerr << "USAGE: " << std::endl;
    std::cerr << "data-generator addresses_filename [generator_id]" << std::endl;
    exit(0);
}

int main(int argc, const char* argv[])
{
    std::vector<Generator::socket_info*> sockets;
    int generator_id = 0;

    if (argc < 2) {
        usage();
    }

    sockets = Generator::LoadAddressesFile(argv[1]);

    if (argc == 3) {
        try {
            generator_id = std::stoi(argv[2]);
        }
        catch(const std::invalid_argument& ia) {
            std::cerr << "Invalid number of threads option '" << argv[2] << "'" << std::endl;
        }
    }

    Generator::GenerateData(sockets, generator_id);

    return 0;
}
