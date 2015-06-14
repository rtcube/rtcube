#pragma once

#include <memory>
#include <string>
#include "../ir/coreimpl.h"
#include "../ir/loadcoreimpl.h"

namespace IR
{
	class DB
	{
		std::unique_ptr<IR::CubeImpl> _cube;

	public:
		DB(std::unique_ptr<IR::CubeImpl> cube): _cube{std::move(cube)} {}

		auto def() -> const IR::CubeDef& {return _cube->def();}

		void insert(const IR::Rows& r) {_cube->insert(r);}
		auto query(const IR::Query& q) -> IR::QueryResult {return _cube->query(q);}
	};

	class Core
	{
		std::unique_ptr<IR::CoreImpl> _core;

	public:
		Core(std::unique_ptr<IR::CoreImpl> core): _core{std::move(core)} {}

		Core(const std::string& type): _core{IR::loadCoreImpl(type)} {}

		auto make_db(const IR::CubeDef& def) -> DB {return DB{std::unique_ptr<IR::CubeImpl>{_core->make_cube(def)}};}
	};
}
