#ifndef RTSERVER_H_
#define RTSERVER_H_

#include "../cudacore/api.h"
#include "../cubesql/cubedef.h"

int RunServers(CudaCore::RTCube &cube, const CubeSQL::CubeDef &def, char *hostaddr_tcp, char *hostaddr_udp);

#endif /* RTSERVER_H_ */
