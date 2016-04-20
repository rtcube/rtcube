#ifndef GENERATOR_H
#define GENERATOR_H

#include <vector>
#include <string>
#include "../row-generator/Generator.h"

namespace Generator
{

void GenerateData(const std::vector<socket_info *> &sockets, int generator_id);

}

#endif // GENERATOR_H
