#pragma once

#include "../ir/IR.h"

namespace DummyCore
{
	class RTCubeP;
	class RTCube
	{
		RTCubeP* p;

		RTCube(const RTCube&); // = delete

	public:
		RTCube(const IR::CubeDef&);
		~RTCube();

#if __cplusplus >= 201103L
		RTCube(RTCube&& o): p(o.p) {o.p = 0;}
#endif

		void insert(const IR::Rows&);
		IR::QueryResult query(const IR::Query&);
	};
}
