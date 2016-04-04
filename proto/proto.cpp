#include "proto.h"

#include <sstream>
#include <iostream>
#include "../util/on_exit.h"

using namespace std;

namespace proto
{
	inline auto read_internal(const string& ss, size_t* startindex) -> std::optional<value>
	{
		uint8_t len;
		if (*startindex == ss.size())
			return nullopt;
		len = ss[*startindex];
		*startindex += 1;
		auto valuestr = ss.substr(*startindex, len);
		auto result = value{valuestr};
		*startindex += len;
		return result;
	}

	auto unserialize(const string& in) -> std::vector<value>
	{
		auto out = std::vector<value>{};
		size_t startindex = 0;
		for(;;)
		{
			auto v = read_internal(in, &startindex);
			if (v)
				out.emplace_back(*v);
			else
				break;
		}
		return out;
	}

	auto write(std::ostream& out, const value& v) -> void
	{
		auto s = char(uint8_t(v.data.size()));
		out.write(&s, 1);
		out.write(v.data.data(), v.data.size());
	}

	auto serialize(const std::vector<value>& in) -> string
	{
		auto full_len = 0;
		for (const auto& v : in)
			full_len += 1 + v.data.size();

		auto out = string{};
		out.reserve(full_len);

		for (const auto& v : in)
		{
			out += char(v.data.size());
			out += v.data;
		}

		return out;
	}

	auto serialize(const std::vector<string>& in) -> string
	{
		auto full_len = 0;
		for (const auto& v : in)
			full_len += 1 + v.size();

		auto out = string{};
		out.reserve(full_len);

		for (const auto& v : in)
		{
			out += char(v.size());
			out += v;
		}

		return out;
	}
}
