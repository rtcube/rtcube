#include "query.h"

#include "../cubesql/parser.h"
#include "../cubesql/tokenizer.h"
#include "../to_ir/to_ir.h"
#include "../util/HostPort.h"

#include <iostream>
#include <system_error>
#include <cassert>
#include <cstring>

namespace RTCube
{
	auto connect(const std::vector<std::string>& hostports) -> std::vector<fd> // sockets
	{
		std::vector<fd> sockets;
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

			sockets.emplace_back(s);
		}
		return sockets;
	}

	auto query(const std::vector<std::string>& hostports, const std::string& cubedef, const std::string& cubesql) -> IR::Cube
	{
		return query(connect(hostports), cubedef, cubesql);
	}

	auto query(const std::vector<fd>& sockets, const std::string& cubedef, const std::string& cubesql) -> IR::Cube
	{
		return query(std::vector<int>{sockets.begin(), sockets.end()}, cubedef, cubesql);
	}

	auto query(const std::vector<int>& sockets, const std::string& cubedef, const std::string& cubesql) -> IR::Cube
	{
		auto d = CubeSQL::parseCubeDef(cubedef);
		auto d_ir = toIR(d);

		auto q = CubeSQL::parse(cubesql);
		auto q_ir = toIR(d, d_ir, q);

		auto r_def = IR::resultCubeDef(d_ir, q_ir);

		if (r_def.dims[0].range < 5)
			throw 0;

		for (auto s : sockets)
			if (::send(s, cubesql.c_str(), cubesql.size() + 1, 0) != cubesql.size() + 1)
				throw std::system_error{errno, std::system_category(), "RTCube::query(): send"};

		auto results = std::vector<IR::Cube>{};

		for (auto s : sockets)
		{
			auto data = std::vector<IR::mea>{};
			data.resize(r_def.cube_size());
			
			int read = 0;
			int buf_size = data.size() * sizeof(IR::mea);
			do
			{
				auto len = ::recv(s, ((char*)data.data()) + read, buf_size - read, 0);
				if (len < 0)
					throw std::system_error{errno, std::system_category(), "RTCube::query(): recv"};
				read += len;
			}
			while (read != buf_size);
			results.emplace_back(r_def, data);
		}

		// TODO merge

		std::cout << "Results size: " << results.size() << std::endl;

		return results[0];
	}
}

void RTCube_free_error(RTCube_Error* e)
{
	delete[] e->message;
	delete e;
}

void RTCube_free_cube(RTCube_Cube* c)
{
	delete[] c->dim_ranges;
	delete[] c->mea_types;
	delete[] c->data;
	delete c;
}

inline void c_throw(std::invalid_argument& e, struct RTCube_Error** error)
{
	if (!error)
		return;

	*error = new RTCube_Error{};
	auto len = strlen(e.what()) + 1;
	(*error)->type = RTCube_Error::InvalidArgument;
	(*error)->code = 0;
	(*error)->message = new char[len];
	memcpy((*error)->message, e.what(), len);
}

inline void c_throw(std::system_error& e, struct RTCube_Error** error)
{
	if (!error)
		return;

	*error = new RTCube_Error{};
	auto len = strlen(e.what()) + 1;
	(*error)->type = RTCube_Error::SystemError;
	(*error)->code = e.code().value();
	(*error)->message = new char[len];
	memcpy((*error)->message, e.what(), len);
}

inline auto c_cube(const IR::Cube& cube) -> RTCube_Cube*
{
	auto c_cube = new RTCube_Cube{};

	c_cube->dim_size = cube.def.dims.size();
	c_cube->dim_ranges = new uint[cube.def.dims.size()];
	for (int i = 0; i < cube.def.dims.size(); ++i)
		c_cube->dim_ranges[i] = cube.def.dims[i].range;

	c_cube->mea_size = cube.def.meas.size();
	c_cube->mea_types = new uint[cube.def.meas.size()];
	for (int i = 0; i < cube.def.meas.size(); ++i)
		c_cube->mea_types[i] = int(cube.def.meas[i].type);

	c_cube->data_size = cube.data.size();
	c_cube->data = new int64_t[cube.data.size()];
	for (int i = 0; i < cube.data.size(); ++i)
		c_cube->data[i] = cube.data[i].i;

	return c_cube;
}

struct RTCube_Cube* RTCube_query(const int* sockets, int sockets_len, const_cstring cubedef, const_cstring cubesql, struct RTCube_Error** error)
{
	try
	{
		return c_cube(RTCube::query(std::vector<int>{sockets, sockets + sockets_len}, cubedef, cubesql));
	}
	catch (std::invalid_argument& e)
	{
		c_throw(e, error);
	}
	catch (std::system_error& e)
	{
		c_throw(e, error);
	}
}

struct RTCube_Cube* RTCube_connect_query(const const_cstring* hostports, int hostports_len, const_cstring cubedef, const_cstring cubesql, struct RTCube_Error** error)
{
	try
	{
		return c_cube(RTCube::query(std::vector<std::string>{hostports, hostports + hostports_len}, cubedef, cubesql));
	}
	catch (std::invalid_argument& e)
	{
		c_throw(e, error);
	}
	catch (std::system_error& e)
	{
		c_throw(e, error);
	}
}
