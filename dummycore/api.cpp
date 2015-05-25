#include "api.h"

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

	void RTCube::insert(const IR::Rows&)
	{

	}

	IR::QueryResult RTCube::query(const IR::Query&)
	{
		return IR::QueryResult();
	}
}
