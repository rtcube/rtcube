#pragma once

namespace IR { class Cube; }
namespace CubeSQL { class CubeDef; }

int RunServers(IR::Cube &cube, const CubeSQL::CubeDef &def, char *hostaddr_tcp, char *hostaddr_udp);
