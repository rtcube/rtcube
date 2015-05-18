#include <iostream>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdexcept>

#include "Generator.h"

using namespace std;

void usage(){
	cerr << "USAGE: " << endl;
	cerr << "RowGenerator ip port [filename | -r number_of_colums]" << endl;
	exit(0);
}

int main(int argc, const char* argv[]){
    bool from_file = false;
	time_t startTime, endTime;
	double seconds;
	int port, no_cols, no_rows;
	const char* ip, * filepath;
	RowGenerator generator = RowGenerator();

	if (argc < 4){
		usage();
	}
	port = std::stoi(argv[2]);
	ip = argv[1];
	if (from_file = (argc == 4)){
		filepath = argv[3];
		if (!generator.LoadCubeFile(filepath)){
			cout << "Couldn't load cube file " << filepath << endl;
			return 1;
		}
		no_cols = generator.NoColumns();
	}else{
		no_cols = std::stoi(argv[4]);
		if (no_cols < 1){
			cerr << "Number of columns must be at least 1" << endl;
			return 3;
		}
		generator.SetNoColumns(no_cols);
	}

	if (!generator.Connect(port,ip)){
		cerr << "Could not connect to " << ip << ":" << port << endl;
		return 2;
	}

	std::string line;
	while(true){
		cout << "%d - generate rows; S|s - send data; Q|q - quit:" << endl;
		getline(cin, line);

		//Q|q - exit
		if(line[0] == 'q' || line[0] == 'Q'){
			return 0;
		}
        //S|s - send the data
        if (line[0] == 's' || line[0] == 'S'){
            time(&startTime);
            int bytes = generator.Send();
            time(&endTime);
            seconds = difftime(endTime, startTime);
            cout << endl << "Sent "<< bytes << " bytes - " << bytes/ seconds / 1024 / 1024 << " Mbytes/s" << endl;
            continue;
        }

		//%d - Generate rows
		try {no_rows = stoi(line);}
		catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << '\n';
            continue;
        }

		time(&startTime);

		if(!from_file){
			generator.GenerateRandom(no_rows);
		}else{
			generator.Generate(no_rows);
		}
		time(&endTime);
		seconds = difftime(endTime, startTime);
		cout << endl << "Generated " << (int)(no_cols * no_rows * (sizeof(int) + 1)/ seconds / 1024 / 1024) << " Mbytes/s" << endl;
	}


	return 0;
}
