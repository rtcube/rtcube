#pragma once

#include <vector>
#include <string>
#include "token.h"
#include "query.h"

namespace CubeSQL
{
	auto parse(const std::vector<token>& data) -> Select;
}
