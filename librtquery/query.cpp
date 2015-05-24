#include "query.h"

#include "../cubesql/parser.h"
#include "../cubesql/tokenizer.h"

#include <iostream>
#include <cassert>
#include <cstring>

namespace RTCube
{
	void query(const std::string& cubesql)
	{
		auto tokens = CubeSQL::tokenize(cubesql);
		auto q = CubeSQL::parse(tokens);
	}
}

void RTCube_free_error(RTCube_error* e)
{
	delete[] e->message;
	delete e;
}

void RTCube_query(const char* cubesql, RTCube_error** error)
{
	try
	{
		RTCube::query(cubesql);
	}
	catch (std::invalid_argument& e)
	{
		if (error)
		{
			*error = new RTCube_error{};
			auto len = strlen(e.what()) + 1;
			(*error)->message = new char[len];
			memcpy((*error)->message, e.what(), len);
		}
	}
}
