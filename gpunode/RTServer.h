#pragma once

namespace IR { class DB; }
namespace CubeSQL { class CubeDef; }

int RunServers(IR::DB &cube, const CubeSQL::CubeDef &def, char *hostaddr_tcp, char *hostaddr_udp);
