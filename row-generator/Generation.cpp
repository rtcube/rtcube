#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <thread>
#include <ctime>

#include "../proto/proto.h"
#include "Send.cpp"

#define ROWS_PER_BLOCK 1000000
#define BUFFER_SIZE 4096
#define ROWS_PER_TIMESTAMP 100

using namespace std;

namespace Generator {

// generates a single column value given a cube_info struct pointer
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

    if (with_time)
        v.emplace_back(time);

    for (int i = 0; i < cube->no_cols; ++i) {
        v.emplace_back(getVal(i, (unsigned int *)rand_r_seed,  cube));
    }

    return proto::serialize(v);
}

bool canGenerateFromCube(cube_info *cube) {
    if (!cube->max_vals || !cube->min_vals || !cube->range_or_list) {
        std::cerr << "Variables for random generation are not set" << std::endl;
        return false;
    }
    return true;
}

// generates no_rows rows based on the given cube_info pointer,
// then sends them in blocks of size smaller than BUFFER_SIZE (4096) over the sockets.
int generateRows(int no_rows, unsigned int * rand_r_seed,  cube_info *cube, std::vector<socket_info*> sockets,
                 bool with_time = true) {

    int time;
    int socket_index = 0;
    int sockets_size = sockets.size();
    int rows_per_send = BUFFER_SIZE / (cube->no_cols + 1) / (sizeof(int) + 1);
    std::string row = "";

    timespec ts;

    int bytes = 0;
    for (int i = 0; i < no_rows; ++i) {
        clock_gettime(CLOCK_REALTIME, &ts);
        time = (int)(ts.tv_sec % 20000 * 1000.0f + ts.tv_nsec * 0.000001f);
        row += generateIntRow(time, rand_r_seed, cube,  with_time);

        if ((i % rows_per_send) == 0) {
            socket_index = (socket_index + 1) % sockets.size();
            bytes += sendRow(sockets[socket_index], row);
            row = "";
        }
    }
    return bytes;
}

// function to pass for a row generating thread
void TGenerate(int thread_nr, int no_blocks, cube_info *cube, std::vector<socket_info*> sockets) {
    unsigned int rand_r_seed = (unsigned int) ((int) time(NULL)) * (thread_nr + 1);
    for(int i=0; i < no_blocks; ++i) {
        timespec ts_start;
        clock_gettime(CLOCK_REALTIME, &ts_start); // Works on Linux

        std::cout << "Thread[" << thread_nr << "] - generating data block no "<< i +1 << std::endl;
        int bytes = generateRows(ROWS_PER_BLOCK, &rand_r_seed, cube, sockets);

        timespec ts_end;
        clock_gettime(CLOCK_REALTIME, &ts_end); // Works on Linux
        long ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0f + ts_end.tv_nsec * 0.000001f - ts_start.tv_nsec * 0.000001f;
        double MBps = (bytes * 0.000001) / (ms * 0.001);
        std::cerr << "Sent " << bytes << " bytes in " << ms << " ms" << " (" << MBps << " MB/s)" << std::endl;
    }
}

// Main function
// Launches no_threads threads, each generating no_blocks * ROWS_PER_BLOCK (1000000) rows
void StartGenerating(int no_blocks, cube_info *cube, std::vector<socket_info*> sockets, int no_threads = 2) {
    if (!canGenerateFromCube(cube)) {
        std::cerr << "Cannot generate from cube definition." << endl;
        return;
    }

    std::thread * genThreads = new thread [no_threads];

    for (int i = 0; i < no_threads; i++) {
        genThreads[i] = std::thread(TGenerate, i, no_blocks, cube, sockets);
    }

    for (int i = 0; i < no_threads; i++) {
        genThreads[i].detach();
    }
}

// cube definition parsing
//
// parses a line of comma separated integer values
void parseIntLine(std::string line, std::vector<int> & vector) {
    std::string elem;
    std::stringstream ss(line);
    int val;

    while (getline(ss, elem, ',')) {
        try {
            val = std::stoi(elem);
            vector.push_back(val);
        }
        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Could not parse '" << elem << "' in line '" << line << std::endl;
        }
    }
}

// returns a pointer to a cube_info struct, parsed from a file
cube_info* LoadCubeFile(std::string filename) {
    std::ifstream file(filename.c_str());
    cube_info* cube = new cube_info();
    if (!file.is_open()) {
        return NULL;
    }

    std::string line;
    int dim_count = 0, m_count = 0;
    // count the number of dimensions and measures
    std::getline(file, line);
    if (line[0] == '#') {
        while (std::getline(file, line) && line[0] != '#') {
            dim_count++;
        }
        while (std::getline(file, line)) {
            m_count++;
        }
    } else {
        // file does not begin with '#'
        return NULL;
    }

    // init the arrays
    int no_cols = dim_count + m_count;
    cube->no_cols = no_cols;
    cube->range_or_list = new bool[no_cols];
    cube->min_vals = new int [no_cols];
    cube->max_vals = new int [no_cols];
    cube->lists = new std::vector<int>[no_cols];

    //now lets read the data
    file.clear();
    file.seekg(0, std::ios::beg);

    int str_index, len, i = 0;
    while (std::getline(file, line)) {
        if (line[0] != '#') {
            if ((cube->range_or_list[i] = (line[0] == '['))) {
                // range of values
                int val;
                std::string val_substr;

                str_index = line.find_first_of(',');
                cube->min_vals[i] = std::stoi(line.substr(1, str_index - 1));
                len = line.find_first_of(']') - str_index - 1;
                val_substr = line.substr(str_index + 1, len);

                try {
                    val = std::stoi(val_substr);
                    cube->max_vals[i] = val - cube->min_vals[i];
                }
                catch(const std::invalid_argument& ia)
                {
                    std::cerr << "Could not parse '" << val_substr << "' in line '" << line << std::endl;
                }
            } else {
                // list of values
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
