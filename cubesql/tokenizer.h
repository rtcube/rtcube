#pragma once

#include <vector>
#include <string>
#include "token.h"

namespace CubeSQL
{
	auto tokenize(const std::string& data) -> std::vector<token>;
}
