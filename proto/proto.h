#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <cxxcompat/optional>

namespace proto
{
	auto read(std::istream& in) -> std::optional<std::string>;
	auto parse(const std::string& in) -> std::vector<std::string>;

	auto write(std::ostream& out, const std::string&) -> void;
	auto serialize(const std::vector<std::string>&) -> std::string;

	template <typename T>
	auto decode(const std::string&) -> T;

	inline auto encode(const std::string& s) -> std::string {return s;}
	template <>
	inline auto decode<std::string>(const std::string& s) -> std::string {return s;}

	auto encode(int) -> std::string;
	template <>
	auto decode<int>(const std::string&) -> int;

	auto encode(float) -> std::string;
	template <>
	auto decode<float>(const std::string&) -> float;

	inline auto serialize_internal(std::string i)
	{
		return std::move(i);
	}

	template <typename F, typename... R>
	inline auto serialize_internal(std::string i, F first, R... rest)
	{
		auto v = encode(first);
		i += char(v.size());
		i += v;
		return serialize_internal(std::move(i), rest...);
	}

	template <typename... T>
	inline auto serialize(const T&... arg) -> std::string
	{
		return serialize_internal({}, arg...);
	}

	template <int i, typename T>
	inline auto unserialize_internal(std::istream& in, T t) -> T
	{
		return std::move(t);
	}

	template <int i, typename T, typename F, typename... R>
	inline auto unserialize_internal(std::istream& in, T t) -> T
	{
		auto v = read(in);
		std::get<i>(t) = decode<F>(*v);
		return unserialize_internal<i+1, T, R...>(in, std::move(t));
	}

	template <typename... T>
	auto unserialize(const std::string& in) -> std::tuple<T...>
	{
		std::stringstream ss{in, std::ios_base::in};
		ss.exceptions(std::istream::failbit | std::istream::badbit);
		return unserialize_internal<0, std::tuple<T...>, T...>(ss, {});
	}
}
