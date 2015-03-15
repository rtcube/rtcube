#include "proto.h"

#include <sstream>
#include <iostream>
#include "../util/on_exit.h"

using namespace std;

namespace proto
{
	inline auto read_internal(std::istream& ss) -> std::optional<value>
	{
		uint8_t len;
		try
		{
			ss.read((char*) &len, 1);
		}
		catch (istream::failure)
		{
			return nullopt;
		}

		char buf[len];
		ss.read(buf, len);
		return value{string{(char*) buf, len}};
	}

	auto read(std::istream& ss) -> std::optional<value>
	{
		auto mask = ss.exceptions();
		ON_EXIT(x, ss.exceptions(mask);)
		ss.exceptions(std::istream::failbit | std::istream::badbit);

		return read_internal(ss);
	}

	auto parse(const string& in) -> std::vector<value>
	{
		auto out = std::vector<value>{};
		stringstream ss{in, ios_base::in};
		ss.exceptions(std::istream::failbit | std::istream::badbit);
		for(;;)
		{
			auto v = read_internal(ss);
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
