#include <iostream>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdexcept>

#include "Generator.cpp"

using namespace std;

void usage(){
	cerr << "USAGE: " << endl;
	cerr << "RowGenerator ip port filename" << endl;
	exit(0);
}

int main(int argc, const char* argv[]){
	Generator::cube_info * cube;
	int port, no_cols, no_rows;
	const char* ip, * filepath;
	//RowGenerator generator = RowGenerator();

	if (argc < 4){
		usage();
	}
	port = std::stoi(argv[2]);
	ip = argv[1];
    filepath = argv[3];
    if (!(cube = Generator::LoadCubeFile(filepath))){
        cout << "Couldn't load cube file " << filepath << endl;
        return 1;
    }

	std::string line;
	while(true){
		cout << "%d - generate %d blocks; Q|q - quit:" << endl;
		getline(cin, line);

		//Q|q - exit
		if(line[0] == 'q' || line[0] == 'Q'){
			return 0;
		}

		//%d - Generate rows
		try {no_rows = stoi(line);}
		catch (const std::invalid_argument& ia) {
            //std::cerr << "Invalid argument: " << ia.what() << '\n';
            continue;
        }
        Generator::StartGenerating(no_rows,cube, ip, port);
	}
	return 0;
}
