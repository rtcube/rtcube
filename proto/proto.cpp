#include "proto.h"

#include <sstream>
#include <iostream>
#include "../util/on_exit.h"

using namespace std;

namespace proto
{
	inline auto read_internal(std::istream& ss) -> std::optional<std::string>
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
		return string{(char*) buf, len};
	}

	auto read(std::istream& ss) -> std::optional<std::string>
	{
		auto mask = ss.exceptions();
		ON_EXIT(x, ss.exceptions(mask);)
		ss.exceptions(std::istream::failbit | std::istream::badbit);

		return read_internal(ss);
	}

	auto parse(const std::string& in) -> std::vector<std::string>
	{
		auto out = std::vector<std::string>{};
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

	auto write(std::ostream& out, const std::string& v) -> void
	{
		auto s = char(uint8_t(v.size()));
		out.write(&s, 1);
		out.write(v.data(), v.size());
	}

	auto serialize(const std::vector<std::string>& in) -> std::string
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

	auto encode(int v) -> std::string
	{
		return {(char*) &v, 4};
	}

	template <>
	auto decode<int>(const std::string& v) -> int
	{
		return *(int*) v.data();
	}
}
