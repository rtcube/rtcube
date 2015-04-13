#pragma once

#include <string>

namespace CubeSQL
{
	struct token
	{
		std::string code;

		inline token() {}
		inline token(const std::string& code): code(code) {}
	};
}
