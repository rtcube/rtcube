#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct RTCube_Error
{
	enum Type { InvalidArgument, SystemError } type;
	int code;
	char* message;
};
void RTCube_free_error(struct RTCube_Error*);

typedef unsigned int uint;

struct RTCube_Cube
{
	uint dim_size;
	uint* dim_ranges;

	uint mea_size;
	uint* mea_types; // 0 = int64_t, 1 = double

	uint data_size;
	int64_t* data; // int64_t or double!
};
void RTCube_free_cube(struct RTCube_Cube*);

typedef const char* const_cstring;
struct RTCube_Cube* RTCube_query(const int* sockets, int sockets_len, const_cstring cubedef, const_cstring cubesql, struct RTCube_Error**);
struct RTCube_Cube* RTCube_connect_query(const const_cstring* hostports, int hostports_len, const_cstring cubedef, const_cstring cubesql, struct RTCube_Error**);

#ifdef __cplusplus
}

#include "query.hpp"
#endif
