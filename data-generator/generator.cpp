#include "generator.h"

#include <limits>
#include <iostream>

namespace Generator
{

constexpr int DimCount = 100;
constexpr int MaxX = std::numeric_limits<int>::max();
constexpr int MaxY = 10'000;

inline int sendRow(socket_info * socket, string &row)
{
    int bytes = sendto(socket->fd, row.data(), row.size(), 0,
                       (struct sockaddr *)&(socket->sin6), sizeof(socket->sin6));
    if (bytes < 0) {
        perror("Sendto");
        return -1;
    }

    return bytes;
}

inline string GenerateRow(int d1, int d2, int m1, int m2)
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int time = (int)(ts.tv_sec % 20000 * 1000.0f + ts.tv_nsec * 0.000001f);

    return proto::serialize(std::vector<proto::value> {time, d1, d2, m1, m2});
}

int GeneratePackage(int x, int y, int generator_id, const std::vector<socket_info*> &sockets)
{
    static int socket_id = 0;

    string rows;

    for(int i = 0; i < DimCount; ++i)
    {
        rows += GenerateRow(generator_id, i, x, y);
    }

    socket_id = (socket_id + 1) % sockets.size();

    return sendRow(sockets[socket_id], rows);
}

void GenerateData(atomic_bool* should_exit_flag, const std::vector<Generator::socket_info *> &sockets, int generator_id)
{
    timespec ts_start, ts_end;
    clock_gettime(CLOCK_REALTIME, &ts_start);

    for(size_t x = 0; (x < MaxX) && !(*should_exit_flag); ++x)
    {
        int bytes = 0;

        for(size_t y = 0; y < MaxY; ++y)
        {
            bytes += GeneratePackage(x, y, generator_id, sockets);
        }

        clock_gettime(CLOCK_REALTIME, &ts_end);
        long ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0f + ts_end.tv_nsec * 0.000001f - ts_start.tv_nsec * 0.000001f;

        //std::cout << "Iteration " << x << "; Sent " << bytes << " bytes in " << ms << " ms; " << std::endl;
        std::cout << generator_id << ',' << x << ',' << ms << ',' << (unsigned long long int)(ts_end.tv_sec * 1000.0 + ts_end.tv_nsec * 0.000001) << "\n";
    }
}

}
