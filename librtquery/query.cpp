#include "query.h"

#include "../cubesql/parser.h"
#include "../cubesql/tokenizer.h"
#include "../util/HostPort.h"

#include <iostream>
#include <system_error>
#include <cassert>
#include <cstring>

namespace RTCube
{
	auto connect(const std::vector<std::string> hostports) -> std::vector<int> // sockets
	{
		std::vector<int> sockets;
		sockets.reserve(hostports.size());
		for (const auto& hps : hostports)
		{
			auto hp = HostPort{hps};

			auto s = ::socket(PF_INET6, SOCK_STREAM, 0);
			if (s < 0)
				throw std::system_error{errno, std::system_category(), "RTCube::query(): create socket"};

			::sockaddr_in6 sin6;
			::memset(&sin6, 0, sizeof(sin6));
			sin6.sin6_family = AF_INET6;
			::memcpy(&sin6.sin6_addr, hp.ip, 16);
			sin6.sin6_port = hp.port;

			if (::connect(s, (sockaddr*) &sin6, sizeof(sin6)) != 0)
				throw std::system_error{errno, std::system_category(), "RTCube::query(): connect"};

			sockets.push_back(s);
		}
		return sockets;
	}

	void query(const std::vector<int> sockets, const std::string& cubesql)
	{
		auto tokens = CubeSQL::tokenize(cubesql);
		auto q = CubeSQL::parse(tokens);
	}

	void query(const std::vector<std::string> hostports, const std::string& cubesql)
	{
		query(connect(hostports), cubesql);
	}
}

void RTCube_free_error(RTCube_error* e)
{
	delete[] e->message;
	delete e;
}

void RTCube_query(const int* sockets_a, int sockets_len, const_cstring cubesql, struct RTCube_error** error)
{
	try
	{
		RTCube::query(std::vector<int>{sockets_a, sockets_a + sockets_len}, cubesql);
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

void RTCube_connect_query(const const_cstring* hostports, int hostports_len, const_cstring cubesql, struct RTCube_error** error)
{
	try
	{
		RTCube::query(std::vector<std::string>{hostports, hostports + hostports_len}, cubesql);
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
