#pragma once

#include <string>
#include <set>

namespace CubeSQL
{
	using Int = long long;
	using Float = double;
	using String = std::string;

	template <typename T>
	struct Range
	{
		bool left_inclusive;
		bool right_inclusive;
		T left;
		T right;
	};

	template <typename T>
	struct Set
	{
		std::set<T> values;
	};

	template <typename TInt, typename TFloat, typename TString>
	struct TriType
	{
		enum Type
		{
			Empty,
			Int,
			Float,
			String
		} type;

		TInt i;
		TFloat f;
		TString s;

		TriType(): type(Empty) {}
		TriType(TInt v):           type(Int),    i(v) {}
		TriType(TFloat v):         type(Float),  f(v) {}
		TriType(const TString& v): type(String), s(v) {}

		explicit operator bool() const { return type != Empty; }
		explicit operator TInt() const { return i; }
		explicit operator TFloat() const { return f; }
		explicit operator TString() const { return s; }

		TriType& operator=(TInt v)    {type = Int;    i = v;}
		TriType& operator=(TFloat v)  {type = Float;  f = v;}
		TriType& operator=(TString v) {type = String; s = v;}

		bool operator==(TInt v)           const {return type == Int && i == v;}
		bool operator==(TFloat v)         const {return type == Float && f == v;}
		bool operator==(const TString& v) const {return type == String && s == v;}
	};

	using AnyAtom  = TriType<Int, Float, String>;
	using AnyRange = TriType<Range<Int>, Range<Float>, Range<String>>;
	using AnySet   = TriType<Set<Int>, Set<Float>, Set<String>>;
}
