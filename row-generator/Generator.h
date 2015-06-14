#include "../proto/proto.h"
#include "../util/HostPort.h"
#include <vector>

using namespace std;

namespace Generator{

// struct holding info about the cube's structure
struct cube_info {
    int no_cols;
    bool * range_or_list;
    int * max_vals;
    int * min_vals;
    std::vector<int> * lists;
};

struct socket_info {
    int fd;
    struct sockaddr_in6 sin6;
};

std::vector<socket_info*> LoadAddressesFile(std::string filename);

void StartGenerating(int no_blocks, cube_info *cube, std::vector<socket_info*> sockets, int no_threads);

cube_info* LoadCubeFile(std::string filename);

}

