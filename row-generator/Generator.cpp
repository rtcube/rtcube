#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/stat.h>

#include <vector>
#include <list>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include "../proto/proto.h"
#include <thread>
#include <ctime>

#define ROWS_PER_BLOCK 1000000

using namespace std;

// UDP
namespace Generator {
struct socket_info {
    int fd;
    struct sockaddr_in6 sin6;
};

socket_info * makeSocket(int port, std::string address, bool test_connectivity = false) {
    socket_info * new_socket = new socket_info();
    int fd;
    struct sockaddr_in6 sin6;
    memset(&sin6, 0, sizeof(sin6));
    sin6.sin6_family = AF_INET6;
    inet_pton(AF_INET6, address.data(), &(sin6.sin6_addr));
    sin6.sin6_port=htons(port);

    if ((fd = socket(PF_INET6, SOCK_DGRAM, 0)) < 0) {
        perror("Opening socket.");
        exit(1);
    }

    if (test_connectivity) {
        auto v = proto::serialize("hello");
        int res = sendto(fd, v.data(), v.size(), 0,
                         (struct sockaddr *)&sin6,sizeof(sin6));
        if (res != v.size()) {
            perror("Connectivity test failure.");
            exit(2);
        }
    }

    new_socket->fd = fd;
    new_socket->sin6 = sin6;

    return new_socket;
}

inline int sendRow(socket_info * socket, string &row) {
    int bytes = sendto(socket->fd, row.data(), row.size(), 0,
                       (struct sockaddr *)&(socket->sin6),sizeof(socket->sin6));
    if (bytes < 0) {
        perror("Sendto");
        std::cerr << socket->fd << endl;
        return 0;
    }

    return bytes;
}
}

// Generation
namespace Generator {
struct cube_info {
    int no_cols;
    bool * range_or_list;
    int * max_vals;
    int * min_vals;
    std::vector<int> * lists;
};

inline int getVal(int col_nr, unsigned int * rand_r_seed, cube_info * cube) {
    if (cube->range_or_list[col_nr]) {
        //range
        return rand_r(rand_r_seed) % cube->max_vals[col_nr] + cube->min_vals[col_nr];
    } else {
        //list
        std::vector<int>* list = &(cube->lists[col_nr]);
        int index = rand_r(rand_r_seed) % list->size();
        return (*list)[index];
    }
}

inline std::string generateIntRow(int time, unsigned int * rand_r_seed, cube_info *cube, bool with_time = true) {
    int val;
    auto v = std::vector<proto::value> {};
    if (with_time) v.emplace_back(time);

    for (int i = 0; i < cube->no_cols; ++i) {
        v.emplace_back(getVal(i, (unsigned int *)rand_r_seed,  cube));
    }

    return proto::serialize(v);
}

bool canGenerateFromCube(cube_info *cube) {
    if (!cube->max_vals || !cube->min_vals || !cube->range_or_list) {
        cerr << "Variables for random generation are not set" << endl;
        return false;
    }
    return true;
}

int generateRows(int no_rows, unsigned int * rand_r_seed,  cube_info *cube, socket_info *socket,  bool with_time = true) {
    int bytes = 0;
    for (int i = 0; i < no_rows; ++i) {
        auto row = generateIntRow(i % 10, rand_r_seed, cube,  with_time);
        bytes += row.size();
        sendRow(socket, row);
    }
    return bytes;
}

void TGenerate(int thread_nr, int no_blocks, cube_info *cube, socket_info *socket) {
    unsigned int rand_r_seed = (unsigned int) thread_nr;
    for(int i=0; i < no_blocks; ++i) {
        time_t startTime, endTime;
        time(&startTime);
        std::cout << "Thread[" << thread_nr << "] - generating data block no "<< i +1 << std::endl;
        int bytes = generateRows(ROWS_PER_BLOCK, &rand_r_seed, cube, socket);
        time(&endTime);
        int seconds = difftime(endTime, startTime) + 1;
        cout << "generated " << bytes /1024 / 1024/ seconds << "Mb/s (" << bytes << " bytes)" << std::endl;
    }
}

void StartGenerating(int no_blocks, cube_info *cube, std::string ip, int port,
                     int no_threads = 2) {

    if (!canGenerateFromCube(cube)) {
        std::cerr << "Cannot generate from cube definition." << endl;
        return;
    }
    auto socket = makeSocket(port, ip, true);

    srand(time(NULL));

    std::thread * genThreads = new thread [no_threads];

    for (int i = 0; i < no_threads; i++) {
        genThreads[i] = std::thread(TGenerate, i, no_blocks, cube, socket);
    }

    for (int i = 0; i < no_threads; i++) {
        genThreads[i].detach();
    }
}

// cube definition parsing
void parseIntLine(std::string line, std::vector<int> & vector) {
    std::string elem;
    std::stringstream ss(line);

    while (getline(ss, elem, ',')) {
        vector.push_back(std::stoi(elem));
    }
}

cube_info* LoadCubeFile(std::string filename) {
    std::ifstream file(filename.c_str());
    cube_info* cube = new cube_info();
    if (!file.is_open()) {
        return NULL;
    }

    std::string line;
    int dim_count = 0, m_count = 0;
    //count the number of dimensions and measures
    std::getline(file, line);
    if (line[0] == '#') {
        while (std::getline(file, line) && line[0] != '#') {
            dim_count++;
        }
        while (std::getline(file, line)) {
            m_count++;
        }
    } else {
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
    while (std::getline(file, line)) {
        if (line[0] != '#') {
            if ((cube->range_or_list[i] = line[0] == '[')) {
                //range of values
                str_index = line.find_first_of(',');
                cube->min_vals[i] = std::stoi(line.substr(1, str_index - 1));
                len = line.find_first_of(']') - str_index - 1;
                cube->max_vals[i] = std::stoi(line.substr(str_index + 1, len)) - cube->min_vals[i];
            } else {
                //int values
                std::vector<int> list;
                parseIntLine(line, list);
                cube->lists[i] = list;
            }
            ++i;
        }
    }

    return cube;
}
}
