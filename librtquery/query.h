#pragma once

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

typedef const char* const_cstring;
void RTCube_query(const int* sockets, int sockets_len, const_cstring cubesql, struct RTCube_Error**);
void RTCube_connect_query(const const_cstring* hostports, int hostports_len, const_cstring cubesql, struct RTCube_Error**);

#ifdef __cplusplus
}

#include "query.hpp"
#endif
