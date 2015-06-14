#pragma once

#include "../ir/coreimpl.h"

class DummyCube final: public IR::CubeImpl
{
public:
	DummyCube(const IR::CubeDef& def): CubeImpl(def) {}

	void insert(const IR::Rows&) override;
	IR::QueryResult query(const IR::Query&) override;
};

class DummyCore final: public IR::CoreImpl
{
public:
	DummyCube* make_cube(const IR::CubeDef& def) override { return new DummyCube(def); }
};

extern "C" DummyCore* init_core() { return new DummyCore(); }
