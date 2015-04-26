#include "to_ir.h"

using CubeSQL::Range;
using CubeSQL::Set;
using CubeSQL::Int;
using CubeSQL::Float;
using CubeSQL::String;
using std::string;

inline auto char_to_int(char c) -> int
{
	if (c >= '0' && c <= '9')
		return c - '0';
	else if (c >= 'a' && c <= 'z')
		return c - 'a' + 10;
	else if (c >= 'A' && c <= 'Z')
		return c - 'A' + 10;
}

auto toIR(const CubeSQL::CubeDef& cube_sql, const IR::CubeDef& cube_ir, const std::vector<std::vector<proto::value>>& rows) -> IR::Rows
{
	auto ir = IR::Rows{cube_ir.dims.size(), cube_ir.meas.size(), rows.size()};

	auto i = 0;
	for (const auto& row : rows)
	{
		auto r = ir[i++];

		auto j = 0;
		auto k = 0;
		for (const auto& dim : cube_sql.dims)
		{
			if (dim.len > 1)
			{
				auto s = string(row[j++]);
				if (dim.len != s.size())
					throw std::invalid_argument("toIR");

				switch (dim.type)
				{
					case CubeSQL::ColType::IntRange:
					{
						auto rg = Range<Int>(dim.r);
						for (char c : s)
							r.dims()[k++] = char_to_int(c) - rg.left - 1 + int(rg.left_inclusive);
					}
					break;

					case CubeSQL::ColType::Char:
						for (char c : s)
							r.dims()[k++] = c;
					break;

					case CubeSQL::ColType::Set:
						switch (dim.s.type)
						{
							case CubeSQL::Type::Int:
							{
								auto ss = Set<Int>(dim.s);
								for (char c : s)
								{
									auto i = char_to_int(c);
									r.dims()[k++] = ss.index_of(i);
								}
							}
							break;

							case CubeSQL::Type::String:
							{
								auto ss = Set<String>(dim.s);
								for (char c : s)
								{
									r.dims()[k++] = ss.index_of(string{c});
								}
							}
							break;
						};
					break;
				}
			}
			else
			{
				switch (dim.type)
				{
					case CubeSQL::ColType::IntRange:
					{
						auto rg = Range<Int>(dim.r);
						auto i = int(row[j++]);
						r.dims()[k++] = i - rg.left - 1 + int(rg.left_inclusive);
					}
					break;

					case CubeSQL::ColType::Set:
						switch (dim.s.type)
						{
							case CubeSQL::Type::Int:
							{
								auto s = Set<Int>(dim.s);
								auto v = int(row[j++]);
								r.dims()[k++] = s.index_of(v);
							}
							break;

							case CubeSQL::Type::Float:
							{
								auto s = Set<Float>(dim.s);
								auto v = float(row[j++]);
								r.dims()[k++] = s.index_of(v);
							}
							break;

							case CubeSQL::Type::String:
							{
								auto s = Set<String>(dim.s);
								auto v = string(row[j++]);
								r.dims()[k++] = s.index_of(v);
							}
							break;
						};
					break;

					case CubeSQL::ColType::Time:
					{
						auto i = int(row[j++]);
						r.dims()[k++] = i;
						// TODO think about time range, it can work perfectly.
					}
					break;

					case CubeSQL::ColType::Char:
					{
						auto s = string(row[j++]);
						if (s.size() != 1)
							throw std::invalid_argument("toIR");
						r.dims()[k++] = s[0];
					}
					break;
				}
			}
		}

		k = 0;
		for (const auto& mea : cube_sql.meas)
		{
			switch (mea.type)
			{
				case CubeSQL::ColType::Float:
				case CubeSQL::ColType::FloatRange:
					r.meas()[k++].f = float(row[j++]);
				break;

				case CubeSQL::ColType::Int:
				case CubeSQL::ColType::IntRange:
					r.meas()[k++].i = int(row[j++]);
				break;
			}
		}
	}
	
	return ir;
}
