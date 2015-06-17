#include "../proto/proto.h"
#include "../util/HostPort.h"
#include <vector>

using namespace std;

namespace Generator{

enum col_types{
    range_of_vals = 0,
    list_of_vals = 1,
    function_vals = 2
};

struct function_params{
  int period;
  int peak;
};

// struct holding info about the cube's structure
struct cube_info {
    int no_cols;
    col_types * col_type;
    int * max_vals;
    int * min_vals;
    function_params * f_params;
    std::vector<int> * lists;
};

struct socket_info {
    int fd;
    struct sockaddr_in6 sin6;
};

std::vector<socket_info*> LoadAddressesFile(std::string filename);

void StartGenerating(int no_blocks, cube_info *cube, std::vector<socket_info*> sockets, int no_threads);

cube_info* LoadCubeFile(std::string filename);

bool canGenerateFromCube(cube_info *cube);

}

