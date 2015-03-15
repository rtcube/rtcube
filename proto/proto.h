#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <cxxcompat/optional>
#include <stdexcept>

namespace proto
{
	using std::string;

	struct value
	{
		string data;

		inline operator string() const {return data;}
		inline operator    int() const {if (data.size() != 4) throw std::domain_error{  "int(proto::value)"}; return *(  int*)data.data();}
		inline operator  float() const {if (data.size() != 4) throw std::domain_error{"float(proto::value)"}; return *(float*)data.data();}

		inline value(const string& d): data(d) {}
		inline value(const char* d): data(d) {}
		inline value(int v): data((char*) &v, 4) {}
		inline value(float v): data((char*) &v, 4) {}
	};

	inline auto operator==(const value& a, const value& b)  {return a.data == b.data;}
	inline auto operator==(const value& a, const string& b) {return a.data == b;}
	inline auto operator==(const string& a, const value& b) {return a == b.data;}

	auto read(std::istream& in) -> std::optional<value>;
	auto parse(const string& in) -> std::vector<value>;

	auto write(std::ostream& out, const value&) -> void;
	auto serialize(const std::vector<value>&) -> string;
	auto serialize(const std::vector<string>&) -> string;

	inline auto serialize_internal(string i)
	{
		return std::move(i);
	}

	template <typename F, typename... R>
	inline auto serialize_internal(string i, F first, R... rest)
	{
		auto v = value(first);
		i += char(v.data.size());
		i += v.data;
		return serialize_internal(std::move(i), rest...);
	}

	template <typename... T>
	inline auto serialize(const T&... arg) -> string
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
		std::get<i>(t) = F(*v);
		return unserialize_internal<i+1, T, R...>(in, std::move(t));
	}

	template <typename... T>
	auto unserialize(const string& in) -> std::tuple<T...>
	{
		std::stringstream ss{in, std::ios_base::in};
		ss.exceptions(std::istream::failbit | std::istream::badbit);
		return unserialize_internal<0, std::tuple<T...>, T...>(ss, {});
	}
}
