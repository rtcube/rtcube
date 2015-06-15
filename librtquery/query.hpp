#pragma once

#include <vector>
#include <string>
#include <unistd.h>

#include "../ir/IR.h"

namespace RTCube
{
	struct fd
	{
		int _fd = -1;

		fd() {}
		explicit fd(int fd): _fd{fd} {}

		~fd() {if (_fd > 0) close(_fd);}

		fd(const fd&) = delete;
		fd(fd&& o): fd(o._fd) {o._fd = -1;}

		operator int() const {return _fd;}
	};

	// TODO Return something prettier than IR::Cube.
	auto query(const std::vector<std::string>& hostports, const std::string& cubedef, const std::string& cubesql) -> IR::Cube;

	auto connect(const std::vector<std::string>& hostports) -> std::vector<fd>;
	auto query(const std::vector<int>& sockets, const std::string& cubedef, const std::string& cubesql) -> IR::Cube;
	auto query(const std::vector<fd>& sockets, const std::string& cubedef, const std::string& cubesql) -> IR::Cube;
}
