#ifdef _GNU_SOURCE
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sendfile.h>
#endif // _GNU_SOURCE

#include <vector>
#include <list>
#include <queue>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include "../proto/proto.h"
#include <thread>
#include <mutex>
#include <ctime>
#include <condition_variable>

#define ROWS_PER_FILE 1000000
using namespace std;

namespace Generator{
	//socket
	int _fd;
	struct sockaddr_in6 _sin6;

    bool MakeSocket(int port, std::string address){
		memset(&_sin6, 0, sizeof(_sin6));
		_sin6.sin6_family = AF_INET6;
		inet_pton(AF_INET6, address.data(), &(_sin6.sin6_addr));
		_sin6.sin6_port=htons(port);

		if( (_fd=socket(PF_INET6,SOCK_DGRAM,0)) < 0){ perror("Opening socket."); return false;}

        auto v = proto::serialize("hello");
        int res = sendto(_fd, v.data(), v.size(), 0,
            (struct sockaddr *)&_sin6,sizeof(_sin6));

		return res == v.size();
	}

	inline int sendRow(string &row){
		int bytes = sendto(_fd, row.data(), row.size(), 0,
		(struct sockaddr *)&_sin6,sizeof(_sin6));

		return bytes;
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
    int sendVector(std::vector<string> &rows){
        int total_bytes = 0;

        for (auto &row : rows){
            total_bytes += sendRow(row);
        }
        return total_bytes;
    }
    void saveRow(std::string row, std::string filename){
        std::ofstream out;
        out.open(filename, std::ios::app);
        out << row;
        out.close();
	}
}

//Generation
namespace Generator{
    struct cube_info{
        int no_cols;
        bool * range_or_list;
        int * max_vals;
        int * min_vals;
        std::vector<int> * lists;
	};

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

	inline int getVal(int col_nr, cube_info * cube){
		if (cube->range_or_list[col_nr]){
			//range
			return rand() % cube->max_vals[col_nr] + cube->min_vals[col_nr];
		}else{
			//list
			std::vector<int> list = *(&(cube->lists[col_nr]));
			int index = rand() % list.size();
			return list[index];
		}
	}

	inline std::string generateIntRow(int time,cube_info *cube, bool with_time = true){
		int val;
        auto v = std::vector<proto::value>{};
        if (with_time) v.emplace_back(time);

		for (int i = 0; i < cube->no_cols; ++i){
			v.emplace_back(getVal(i, cube));
		}

		return proto::serialize(v);
	}

	bool canGenerateFromCube(cube_info *cube){
		if (!cube->max_vals || !cube->min_vals || !cube->range_or_list){
			cerr << "Variables for random generation are not initialized" << endl;
		}
		return true;
	}

    //threads
    std::list<string> data_files;
    std::queue<vector<string>> data_queue;
    std::mutex list_mutex;
    std::condition_variable cv;
    bool done = false;

// Generates rows based on the loaded file
	void GenerateToFile(int no_rows,cube_info * cube, bool with_time ){
		if (!canGenerateFromCube(cube)){
            std::cerr << "Cannot generate from file" << endl;
			return;
		}
		srand(time(NULL));

        std::string filename =  std::to_string(rand()) + ".data";

        for (int i = 0; i < no_rows; ++i){
            auto row = generateIntRow(i,cube, with_time);
            saveRow(row, filename);
        }

        {
            std::lock_guard<std::mutex> lock(list_mutex);
            std::cout<< filename << std::endl;
            data_files.push_back(filename);
        }
        cv.notify_one();
	}
    int GenerateToMemory(int no_rows, cube_info *cube,  bool with_time = true){
		if (!canGenerateFromCube(cube)){
            std::cerr << "Cannot generate from file" << endl;
			return 0;
		}
		srand(time(NULL));
        std::vector<string> rows;
        int bytes = 0;
        for (int i = 0; i < no_rows; ++i){
            auto row = generateIntRow(i % 10, cube,  with_time);
            rows.push_back(row);
            bytes += row.size();
        }

        {
            std::lock_guard<std::mutex> lock(list_mutex);
            data_queue.push(rows);
        }
        cv.notify_one();
        return bytes;
	}

    void TGenerate(int no_rows, cube_info *cube){
        for(int i=0; i < no_rows; ++i){
            time_t startTime, endTime;
            time(&startTime);
            std::cout<< "generating data block no "<< i +1 << std::endl;
            int bytes = GenerateToMemory(ROWS_PER_FILE, cube);
            time(&endTime);
            int seconds = difftime(endTime, startTime) + 1;
            cout << "generated " << bytes /1024 / 1024/ seconds << "Mb/s (" << bytes << " bytes)" << std::endl;
        }
        std::unique_lock<std::mutex> lk(list_mutex);
        cv.wait(lk, []{return data_files.size() == 0;});
        done = true;
    }
    void TSendFromFiles(){
        while(!done){
            std::unique_lock<std::mutex> lk(list_mutex);
            cv.wait(lk, []{return data_files.size() > 0;});

            auto filename = data_files.front();
            data_files.pop_front();
            lk.unlock();

            cout <<  "sending " << filename << endl;
            time_t startTime, endTime;
            time(&startTime);
            int bytes = sendFile(filename);
            time(&endTime);
            int seconds = difftime(endTime, startTime) + 1;
            cout << "sent " << bytes /1024 / 1024/ seconds << "Mb/s (" << bytes << " bytes)" << std::endl;
            remove(filename.data());
        }
    }
    void TSend(){
        while(!done){
            std::unique_lock<std::mutex> lk(list_mutex);
            cv.wait(lk, []{return !data_queue.empty(); });

            auto rows = data_queue.front();
            data_queue.pop();
            lk.unlock();

            cout <<  "sending " << endl;
            time_t startTime, endTime;
            time(&startTime);
            int bytes = sendVector(rows);
            time(&endTime);
            int seconds = difftime(endTime, startTime) + 1;
            cout << "sent " << bytes /1024 / 1024/ seconds << "Mb/s (" << bytes << " bytes)" << std::endl;
        }
    }

    void StartGenerating(int no_rows, cube_info *cube, std::string ip, int port){
        done = false;
        if(!MakeSocket(port, ip)){
            std::cerr << "Socket error" << std::endl;
            return;
        }
        std::thread genThread(TGenerate, no_rows, cube);
        //std::thread genThread2(TGenerate, no_rows, cube);
        std::thread sendThread(TSend);

        genThread.detach();
        //genThread2.detach();
        sendThread.detach();
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

	cube_info* LoadCubeFile(std::string filename){
		std::ifstream file(filename.c_str());
        cube_info* cube = new cube_info();
		if (!file.is_open()){
			return NULL;
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
			return NULL;
		}

		//init the arrays
		int no_cols = dim_count + m_count;
		cube->no_cols = no_cols;
		cube->range_or_list = new bool[no_cols];
		cube->min_vals = new int [no_cols];
		cube->max_vals = new int [no_cols];
		cube->lists = new vector<int>[no_cols];

		//now lets read the data
		file.clear();
		file.seekg(0, std::ios::beg);

		int str_index, len, i = 0;
		while (std::getline(file, line)){
			if (line[0] != '#'){
				if ((cube->range_or_list[i] = line[0] == '[')){
					//range of values
					str_index = line.find_first_of(',');
					cube->min_vals[i] = std::stoi(line.substr(1, str_index - 1));
					len = line.find_first_of(']') - str_index - 1;
					cube->max_vals[i] = std::stoi(line.substr(str_index + 1, len)) - cube->min_vals[i];
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
						cube->lists[i] = list;
					}
				}
				++i;
			}
		}

		return cube;
	}
}
