#pragma once

#include <vector>
#include <string>
#include <unistd.h>

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

	void query(const std::vector<std::string>& hostports, const std::string& cubesql);

	auto connect(const std::vector<std::string>& hostports) -> std::vector<fd>;
	void query(const std::vector<int>& sockets, const std::string& cubesql);
	void query(const std::vector<fd>& sockets, const std::string& cubesql);
}
