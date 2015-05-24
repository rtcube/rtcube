#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct RTCube_error
{
	char* message;
};
void RTCube_free_error(struct RTCube_error*);

typedef const char* const_cstring;
void RTCube_query(const int* sockets, int sockets_len, const_cstring cubesql, struct RTCube_error**);
void RTCube_connect_query(const const_cstring* hostports, int hostports_len, const_cstring cubesql, struct RTCube_error**);

#ifdef __cplusplus
}

#include "query.hpp"
#endif
