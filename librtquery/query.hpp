#pragma once

#include <vector>
#include <string>

namespace RTCube
{
	void query(const std::vector<std::string> hostports, const std::string& cubesql);

	auto connect(const std::vector<std::string> hostports) -> std::vector<int>; // sockets
	void query(const std::vector<int> sockets, const std::string& cubesql);
}
