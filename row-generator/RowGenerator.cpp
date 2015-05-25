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
    cerr << "USAGE: " << endl;
    cerr << "row-generator ip:port cube_filename [no_threads]" << endl;
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
        cerr << "Couldn't load cube file " << cube_filepath << endl;
        return 1;
    }
    if (argc == 4) {
        no_threads = std::stoi(argv[3]);
    }

    std::string line;
    while(true) {
        cout << "%d - generate %d blocks; i - status request; r - query test; Q|q - quit:" << endl;
        getline(cin, line);

        //Q|q - exit
        if(line[0] == 'q' || line[0] == 'Q') {
            return 0;
        }

        // i - status
		if(line[0] == 'i'){
			if (Generator::StatusRequest(sockets))
				cout << endl << "Sent status request" << endl;
			else
				cout << endl << "Status request error" << endl;
			continue;
		}
		// r - query test
        if(line[0] == 'r'){
			if (Generator::QueryTest(sockets))
				cout << endl << "Sent query request" << endl;
			else
				cout << endl << "Query request error" << endl;
			continue;
 		}

        //%d - Generate rows
        try {
            no_blocks = stoi(line);
        }
        catch (const std::invalid_argument& ia) {
            //std::cerr << "Invalid argument: " << ia.what() << '\n';
            continue;
        }
        Generator::StartGenerating(no_blocks,cube, sockets, no_threads);
    }
    return 0;
}
