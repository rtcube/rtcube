#pragma once

#include <vector>
#include <string>
#include "token.h"

namespace CubeSQL
{
	auto tokenize(std::string data) -> std::vector<token>;
}
