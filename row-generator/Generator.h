#ifdef _GNU_SOURCE
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sendfile.h>
#endif // _GNU_SOURCE

#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include "../proto/proto.h"

#define DATA_FILENAME "cube.data"
using namespace std;
class RowGenerator{
private:
	int _no_cols;
	bool _connected;
	bool * _range_or_list;
	int * _max_vals;
	int * _min_vals;
	std::vector<int> * _lists;

	//socket
	int _fd;
	struct sockaddr_in6 _sin6;

	inline int sendRow(string row){
		int bytes = sendto(_fd, row.data(), row.size(), 0,
		(struct sockaddr *)&_sin6,sizeof(_sin6));

		return bytes;
	}
	int OLDsendFile(std::string filename){
		int fd;
		FILE *f;
		struct stat st;

		if ((f = fopen(filename.data(), "r")) == NULL){
			cerr << "Unable to open file" << endl;
			return -1;
		}
		if ((fd = fileno(f))<0) {perror("fileno"); return -1;}
		//get the filesize
		fstat(fd, &st);

		return sendfile(_fd, fd, NULL, st.st_size);
	}
    int sendFile(std::string filename){
        int total_bytes = 0;
        std::string row;

        ifstream files;
        files.open(filename.data());

        while(std::getline(files, row)){
            total_bytes += sendRow(row);
        }
        return total_bytes;
    }
	void saveRow(std::string row){
        std::ofstream out;
        out.open(DATA_FILENAME, std::ios::app);
        out << row;
        out.close();
	}

	inline int * getRandomMaxVals(int no_cols){
		int * max_vals = new int [no_cols];

		srand(time(NULL));
		for (int i=0; i<no_cols; ++i){
			max_vals[i] = rand();
		}

		return max_vals;
	}
	inline std::string generateRandomRow(int * max_vals, int no_cols, int time){
		auto v = std::vector<proto::value>{time};

		for (int i = 0; i < no_cols; ++i){
			v.emplace_back(rand() % max_vals[i]);
		}
        std::string s = proto::serialize(v);
		return s;
	}

	inline int getVal(int col_nr){
		if (_range_or_list[col_nr]){
			//range
			return rand() % _max_vals[col_nr] + _min_vals[col_nr];
		}else{
			//list
			std::vector<int> list = _lists[col_nr];
			int index = rand() % list.size();
			return list[index];
		}
	}

	inline std::string generateIntRow(int time, bool with_time = true){
		int val;
        auto v = std::vector<proto::value>{};
        if (with_time) v.emplace_back(time);

		for (int i = 0; i < _no_cols; ++i){
			v.emplace_back(getVal(i));
		}

		return proto::serialize(v);
	}

	void parseStringLine(std::string line, std::vector<std::string> & vector){
		std::string elem;
		std::stringstream ss(line);

		while (getline(ss, elem, ',')){
			vector.push_back(elem);
		}
	}
	void parseIntLine(std::string line, std::vector<int> & vector){
		std::string elem;
		std::stringstream ss(line);

		while (getline(ss, elem, ',')){
			vector.push_back(std::stoi(elem));
		}
	}
	bool canGenerateFromFile(){
		if (!_connected){
			cerr << "Not connected to a socket" << endl;
			return false;
		}
		if (!_max_vals || !_min_vals || !_range_or_list){
			cerr << "Variables for random generation are not initialized" << endl;
		}
		return true;
	}
public:
	// Generates rows with random ranges (no file needed)
	void GenerateRandom(int no_rows, int no_cols){
		SetNoColumns(no_cols);
		GenerateRandom(no_rows);
	}
	void GenerateRandom(int no_rows){
		if (_no_cols == 0){
			//TODO: error handling
			return;
		}
		int* max_vals  = getRandomMaxVals(_no_cols);
		int row_size = _no_cols * (sizeof(int) + 1);

		for (int i = 0; i < no_rows; ++i){
			auto row = generateRandomRow(max_vals, _no_cols, i);
			sendRow(row);
		}
		delete [] max_vals;
	}
	// Generates rows based on the loaded file
	void Generate(int no_rows, bool with_time = true){
		if (!canGenerateFromFile()){
            std::cerr << "Cannot generate from file" << endl;
			return;
		}
		srand(time(NULL));

        for (int i = 0; i < no_rows; ++i){
            auto row = generateIntRow(i, with_time);
            saveRow(row);
        }
	}

	bool LoadCubeFile(std::string filename){
		std::ifstream file(filename.c_str());

		if (!file.is_open()){
			return false;
		}

		std::string line;
		int dim_count = 0, m_count = 0;
		//count the number of dimensions and measures
		std::getline(file, line);
		if (line[0] == '#'){
			while (std::getline(file, line) && line[0] != '#'){
				dim_count++;
			}
			while (std::getline(file, line)){
				m_count++;
			}
		}else{
			//file does not begin with '#'
			return false;
		}

		//init the arrays
		_no_cols = dim_count + m_count;
		_range_or_list = new bool[_no_cols];
		_min_vals = new int [_no_cols];
		_max_vals = new int [_no_cols];
		_lists = new vector<int>[_no_cols];

		//now lets read the data
		file.clear();
		file.seekg(0, std::ios::beg);

		int str_index, len, i = 0;
		while (std::getline(file, line)){
			if (line[0] != '#'){
				if ((_range_or_list[i] = line[0] == '[')){
					//range of values
					str_index = line.find_first_of(',');
					_min_vals[i] = std::stoi(line.substr(1, str_index - 1));
					len = line.find_first_of(']') - str_index - 1;
					_max_vals[i] = std::stoi(line.substr(str_index + 1, len)) - _min_vals[i];
				}else {
					//list of values
					if (line[0] == '"'){
						//string values
						std::vector<std::string> list;
						parseStringLine(line, list);
					}else{
						//int values
						std::vector<int> list;
						parseIntLine(line, list);
						_lists[i] = list;
					}
				}
				++i;
			}
		}

		return true;
	}

	bool Connect(int port, std::string address){
		memset(&_sin6, 0, sizeof(_sin6));
		_sin6.sin6_family = AF_INET6;
		inet_pton(AF_INET6, address.data(), &(_sin6.sin6_addr));
		_sin6.sin6_port=htons(port);

		if( (_fd=socket(PF_INET6,SOCK_DGRAM,0)) < 0){ perror("Opening socket."); return false;}

        auto v = proto::serialize("hello");
        int res = sendto(_fd, v.data(), v.size(), 0,
            (struct sockaddr *)&_sin6,sizeof(_sin6));

		return _connected = (res == v.size());
	}
	bool IsConnected() { return _connected; }
	int Send(char * ip, int port){
		if (!Connect(port, ip))
            return -1;
		return Send();
	}
	int Send(){ return sendFile(DATA_FILENAME); }

	int NoColumns(){ return _no_cols; }
	void SetNoColumns(int no_cols){	_no_cols = no_cols;	}
	RowGenerator(){
	//remove the previous data file if exists
		remove(DATA_FILENAME);}
	~RowGenerator(){
		close(_fd);

		remove(DATA_FILENAME);
		if (_range_or_list)
            delete [] _range_or_list;
		if (_min_vals)
            delete [] _min_vals;
        if(_max_vals)
            delete [] _max_vals;
        if(_lists)
            delete [] _lists;
	}
};
