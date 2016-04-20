#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <thread>
#include <ctime>
#include <cmath>

#include "../proto/proto.h"
#include "Send.cpp"

#define ROWS_PER_BLOCK 1000000
#define BUFFER_SIZE 4096
#define ROWS_PER_TIMESTAMP 100

using namespace std;

namespace Generator {

// generates a single column value given a cube_info struct pointer
inline int getVal(int time, int col_nr, unsigned int * rand_r_seed, cube_info * cube) {
    int val;
    switch (cube->col_type[col_nr])
    {
        case Generator::range_of_vals:
        {
            val = rand_r(rand_r_seed) % cube->max_vals[col_nr] + cube->min_vals[col_nr];
            break;
        }
        case Generator::list_of_vals:
        {
            std::vector<int>* list = &(cube->lists[col_nr]);
            int index = rand_r(rand_r_seed) % list->size();
            val = (*list)[index];
            break;
        }
        case Generator::function_vals:
        {
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            float sin_time = (float) ts.tv_nsec / cube->f_params[col_nr].period;
            val = (int)(std::sin(sin_time) * cube->f_params[col_nr].peak);
            break;
        }
    }

    return val;
}

inline std::string generateIntRow(int time, unsigned int * rand_r_seed, cube_info *cube, bool with_time = true) {
    auto v = std::vector<proto::value> {};

    if (with_time)
        v.emplace_back(time);

    for (int col_nr = 0; col_nr < cube->no_cols; ++col_nr) {
        auto val = getVal(time, col_nr, rand_r_seed, cube);
        v.emplace_back(val);
    }

    return proto::serialize(v);
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


}
