#pragma once

#include "../cubesql/cubedef.h"
#include "../proto/proto.h"
#include "../ir/IR.h"

auto toIR(const CubeSQL::CubeDef&) -> IR::CubeDef;
auto toIR(const CubeSQL::CubeDef&, const IR::CubeDef&, const std::vector<std::vector<proto::value>>&) -> IR::Rows;
