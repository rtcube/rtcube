#include "api.h"

#include <cassert>

namespace DummyCore
{
	struct RTCubeP
	{
		IR::CubeDef def;
	};

	RTCube::RTCube(const IR::CubeDef& d)
		: p(new RTCubeP{d})
	{}

	RTCube::~RTCube()
	{
		delete p;
	}

	void RTCube::insert(const IR::Rows& rows)
	{
		assert(rows.num_dims == p->def.dims.size());
		assert(rows.num_meas == p->def.meas.size());
	}

	IR::QueryResult RTCube::query(const IR::Query&)
	{
		return IR::QueryResult();
	}
}
