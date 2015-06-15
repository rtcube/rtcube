#pragma once

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
