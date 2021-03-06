#pragma once

#include <vector>
#include <string>
#include "token.h"
#include "query.h"
#include "cubedef.h"

namespace CubeSQL
{
	auto parse(const std::vector<token>& data) -> Select;
	auto parseCubeDef(const std::vector<token>& data) -> CubeDef;

	auto parse(const std::string& data) -> Select;
	auto parseCubeDef(const std::string& data) -> CubeDef;
}
