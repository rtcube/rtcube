#pragma once

#include "../ir/IR.h"

namespace CudaCore
{
	class RTCubeP;
	class RTCube
	{
		RTCubeP* p;

		RTCube(const RTCube&); // = delete

	public:
		RTCube(IR::CubeDef);
		~RTCube();

#if __cplusplus >= 201103L
		RTCube(RTCube&& o): p(o.p) {o.p = 0;}
#endif

		void insert(IR::Rows);
		IR::QueryResult query(IR::Query);
	};
}
