#pragma once

#include "../ir/coreimpl.h"
#include "RTCube.cuh"

class CudaCube: public IR::CubeImpl
{
	::RTCube cube;

public:
	CudaCube(const IR::CubeDef& def);
	~CudaCube();

	void insert(const IR::Rows&);
	IR::QueryResult query(const IR::Query&);
};

class CudaCore: public IR::CoreImpl
{
public:
	CudaCube* make_cube(const IR::CubeDef& def) { return new CudaCube(def); }
};

extern "C" CudaCore* init_core() { return new CudaCore(); }
