#pragma once

#include "IR.h"

namespace IR
{
	class CubeImpl
	{
		IR::CubeDef _def;

		CubeImpl(const CubeImpl&); // = delete

	protected:
		CubeImpl(const IR::CubeDef& def): _def(def) {}

	public:
		const IR::CubeDef& def() {return _def;}

		virtual ~CubeImpl() {}

		virtual void insert(const IR::Rows&) = 0;
		virtual IR::QueryResult query(const IR::Query&) = 0;
	};

	class CoreImpl
	{
		CoreImpl(const CoreImpl&); // = delete

	protected:
		CoreImpl() {}

	public:
		virtual ~CoreImpl() {}

		virtual CubeImpl* make_cube(const IR::CubeDef& def) = 0;
	};
}
