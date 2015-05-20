#include "to_ir.h"

using CubeSQL::Range;
using CubeSQL::Set;
using CubeSQL::Int;
using CubeSQL::Float;
using CubeSQL::String;

auto toIR(const CubeSQL::CubeDef& cube) -> IR::CubeDef
{
	auto ir = IR::CubeDef{};
	for (const auto& dim : cube.dims)
	{
		auto d = IR::Dim{};

		switch (dim.type)
		{
			case CubeSQL::ColType::IntRange:
			{
				auto r = Range<Int>(dim.r);
				d.range = r.right - r.left - 1 + int(r.left_inclusive) + int(r.right_inclusive);
			}
			break;

			case CubeSQL::ColType::Set:
				switch (dim.s.type)
				{
					case CubeSQL::Type::Int:
					{
						auto s = Set<Int>(dim.s);
						d.range = s.values.size();
					}
					break;

					case CubeSQL::Type::Float:
					{
						auto s = Set<Float>(dim.s);
						d.range = s.values.size();
					}
					break;

					case CubeSQL::Type::String:
					{
						auto s = Set<String>(dim.s);
						d.range = s.values.size();
					}
					break;
				};
			break;

			case CubeSQL::ColType::Time:
				d.range = 0; // automatic
				if (dim.len != 1)
					throw std::invalid_argument("toIR");
			break;

			case CubeSQL::ColType::Char:
				d.range = 256;
			break;

			default:
				throw std::invalid_argument("toIR");
		}

		for (int i = 0; i < dim.len; ++i)
			ir.dims.push_back(d);
	}

	for (const auto& mea : cube.meas)
	{
		auto m = IR::Mea{};

		switch (mea.type)
		{
			case CubeSQL::ColType::Float:
			case CubeSQL::ColType::FloatRange:
				m.type = IR::Mea::Float;
			break;

			case CubeSQL::ColType::Int:
			case CubeSQL::ColType::IntRange:
				m.type = IR::Mea::Int;
			break;

			default:
				throw std::invalid_argument("toIR");
		}

		ir.meas.push_back(m);
	}
	return ir;
}
