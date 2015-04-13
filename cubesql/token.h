#pragma once

#include "types.h"
#include <string>

namespace CubeSQL
{
	struct token
	{
		std::string code;
		AnyAtom val;

		inline token() {}
		inline token(const std::string& code): code(code) {}
		inline token(const char* code): code(code) {}
	};
}
