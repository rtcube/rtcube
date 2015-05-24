#pragma once

#include <string>

namespace RTCube
{
	void query(const std::string& cubesql);
}

extern "C"
{
	struct RTCube_error
	{
		char* message;
	};
	void RTCube_free_error(struct RTCube_error*);

	void RTCube_query(const char* cubesql, struct RTCube_error**);
}
