#pragma once

#include <string>
#include <set>
#include <algorithm>

namespace CubeSQL
{
	using Int = long long;
	using Float = double;
	using String = std::string;

	enum class Type
	{
		Empty,
		Int,
		Float,
		String
	};

	template <typename T>
	class typeOf;

	template <>
	class typeOf<Int>: public std::integral_constant<Type, Type::Int> {};

	template <>
	class typeOf<Float>: public std::integral_constant<Type, Type::Float> {};

	template <>
	class typeOf<String>: public std::integral_constant<Type, Type::String> {};

	template <typename T>
	struct Range
	{
		bool left_inclusive;
		bool right_inclusive;
		T left;
		T right;
		Range() {}
		Range(bool left_inclusive, bool right_inclusive, T left, T right): left_inclusive(left_inclusive), right_inclusive(right_inclusive), left(left), right(right) {}

		auto index_of(T v) const -> int { return v - left - 1 + int(left_inclusive); }
	};

	template <typename T>
	inline bool operator==(const Range<T>& a, const Range<T>& b)
	{
		return a.left_inclusive == b.left_inclusive && a.right_inclusive == b.right_inclusive && a.left == b.left && a.right == b.right;
	}

	template <typename T>
	inline bool operator!=(const Range<T>& a, const Range<T>& b)
	{
		return !(a == b);
	}

	template <typename T>
	struct Set
	{
		std::vector<T> values;

		Set() {}
		Set(std::initializer_list<T> v): values(v) {}

		auto index_of(T v) const -> int { auto p = std::find(values.begin(), values.end(), v); return p == values.end() ? -1 : p - values.begin(); }
	};

	template <typename T>
	inline bool operator==(const Set<T>& a, const Set<T>& b)
	{
		return a.values == b.values;
	}

	template <typename T>
	inline bool operator!=(const Set<T>& a, const Set<T>& b)
	{
		return !(a == b);
	}

	template <typename TInt, typename TFloat, typename TString>
	struct TriType
	{
		Type type;

		TInt i;
		TFloat f;
		TString s;

		TriType(): type(Type::Empty) {}
		TriType(TInt v):           type(Type::Int),    i(v) {}
		TriType(TFloat v):         type(Type::Float),  f(v) {}
		TriType(const TString& v): type(Type::String), s(v) {}

		explicit operator bool() const { return type != Type::Empty; }
		explicit operator TInt() const { return i; }
		explicit operator TFloat() const { return f; }
		explicit operator TString() const { return s; }

		TriType& operator=(TInt v)    {type = Type::Int;    i = v;}
		TriType& operator=(TFloat v)  {type = Type::Float;  f = v;}
		TriType& operator=(TString v) {type = Type::String; s = v;}

		bool operator==(TInt v)           const {return type == Type::Int && i == v;}
		bool operator==(TFloat v)         const {return type == Type::Float && f == v;}
		bool operator==(const TString& v) const {return type == Type::String && s == v;}
	};

	using AnyAtom  = TriType<Int, Float, String>;
	using AnyRange = TriType<Range<Int>, Range<Float>, Range<String>>;
	using AnySet   = TriType<Set<Int>, Set<Float>, Set<String>>;
}
