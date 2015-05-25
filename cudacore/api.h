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

		void insert(IR::Rows);
		IR::QueryResult query(IR::Query);
	};
}
