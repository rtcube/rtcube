#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>

#include "../proto/proto.h"
#include "../util/HostPort.h"
#include "Generator.h"

using namespace std;

namespace Generator{

socket_info * makeSocket(HostPort dest, bool test_connectivity = false) {
    socket_info * new_socket = new socket_info();
    int fd;
    struct sockaddr_in6 sin6;
    ::memset(&sin6, 0, sizeof(sin6));
    sin6.sin6_family = AF_INET6;
    ::memcpy(&sin6.sin6_addr, dest.ip, 16);
    sin6.sin6_port = dest.port;

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
        return -1;
    }

    return bytes;
}

std::vector<socket_info*> LoadAddressesFile(std::string filename) {
    std::vector<socket_info*> sockets;

    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Unable to open addresses file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line)) {
        auto socket = makeSocket(HostPort(line));
        sockets.push_back(socket);
    }

    return sockets;
}

}
