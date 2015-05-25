#include "api.h"

#include "RTCube.cuh"

namespace CudaCore
{
	class RTCubeP
	{
		::RTCube cube;
	};

	RTCube::RTCube(IR::CubeDef)
		: p(new RTCubeP())
	{}

	RTCube::~RTCube()
	{
		delete p;
	}

	void RTCube::insert(IR::Rows)
	{

	}

	IR::QueryResult RTCube::query(IR::Query)
	{
		return IR::QueryResult();
	}
}
