#include "parser.h"
#include <stdexcept>

using std::string;
using std::move;

namespace CubeSQL {

bool iequals(const string& a, const string& b)
{
	if (b.size() != a.size())
		return false;
	for (size_t i = 0; i < a.size(); ++i)
		if (tolower(a[i]) != tolower(b[i]))
			return false;
	return true;
}

struct token_stream
{
	const std::vector<token>& tokens;
	int i;

	auto is_end()
	{
		return i >= tokens.size();
	}

	auto match_end()
	{
		if (!is_end())
			throw std::invalid_argument("CubeSQL::parse");
	}

	auto match(const char* code)
	{
		if (is_end() || !iequals(tokens[i].code, code))
			throw std::invalid_argument("CubeSQL::parse");
		++i;
	}

	auto try_match(const char* code)
	{
		if (is_end() || !iequals(tokens[i].code, code))
			return false;
		++i;
	}

	auto is(const char* code)
	{
		return !is_end() && iequals(tokens[i].code, code);
	}

	auto readLabel() -> std::string
	{
		if (is_end())
			throw std::invalid_argument("CubeSQL::parse");
		return tokens[i++].code;
	}

	template <typename T>
	auto is() -> bool;

	template <typename T>
	auto read() -> T;

	token_stream(const std::vector<token>& tokens): tokens(tokens), i(0) {}
};

template<>
auto token_stream::is<AnyAtom>() -> bool
{
	return bool(tokens[i].val);
}

template<>
auto token_stream::read<AnyAtom>() -> AnyAtom
{
	auto v = tokens[i++].val;
	if (!v)
		throw std::invalid_argument("CubeSQL::parse");
	return v;
}

template<>
auto token_stream::is<AnyRange>() -> bool
{
	return is("(") || is("<");
}

template<>
auto token_stream::read<AnyRange>() -> AnyRange
{
	bool left_inclusive;
	if (try_match("("))
		left_inclusive = false;
	else if (try_match("<"))
		left_inclusive = true;
	else
		throw std::invalid_argument("CubeSQL::parse");

	auto left = read<AnyAtom>();
	match(",");
	auto right = read<AnyAtom>();

	bool right_inclusive;
	if (try_match(")"))
		right_inclusive = false;
	else if (try_match(">"))
		right_inclusive = true;
	else
		throw std::invalid_argument("CubeSQL::parse");

	if (left.type != right.type)
		throw std::invalid_argument("CubeSQL::parse");

	switch (left.type)
	{
	case Type::Int:
		return Range<Int>{left_inclusive, right_inclusive, Int(left), Int(right)};

	case Type::Float:
		return Range<Float>{left_inclusive, right_inclusive, Float(left), Float(right)};

	case Type::String:
		return Range<String>{left_inclusive, right_inclusive, String(left), String(right)};
	}
}

template<>
auto token_stream::is<AnySet>() -> bool
{
	return is("{");
}

template <typename T>
inline void read_rest(Set<T>& set, token_stream& t)
{
	while (t.try_match(","))
	{
		auto a = t.read<AnyAtom>();
		if (typeOf<T>() != a.type)
			throw std::invalid_argument("CubeSQL::parse");
		set.values.push_back(T(a));
	}
}

template<>
auto token_stream::read<AnySet>() -> AnySet
{
	match("{");

	auto first = read<AnyAtom>();

	switch (first.type)
	{
		case Type::Int:
		{
			auto set = Set<Int>{Int(first)};
			read_rest(set, *this);
			match("}");
			return set;
		}
		case Type::Float:
		{
			auto set = Set<Float>{Float(first)};
			read_rest(set, *this);
			match("}");
			return set;
		}
		case Type::String:
		{
			auto set = Set<String>{String(first)};
			read_rest(set, *this);
			match("}");
			return set;
		}
		default:
			throw std::invalid_argument("CubeSQL::parse");
	}
}

template<>
auto token_stream::is<int>() -> bool
{
	return tokens[i].val.type == Type::Int;
}

template<>
auto token_stream::read<int>() -> int
{
	if (!is<int>())
		throw std::invalid_argument("CubeSQL::parse");
	return Int(tokens[i++].val);
}

// Expr ::= FieldName | Constant | Operation '(' Expr ( ',' Expr )* ')'
auto readExpr(token_stream& t) -> AnyExpr
{
	if (t.is<AnyAtom>())
		return std::make_unique<ConstantExpr>(t.read<AnyAtom>());

	auto name = t.readLabel();

	if (t.try_match("("))
	{
		auto expr = std::make_unique<OperationExpr>(move(name));
		do
		{
			expr->args.push_back(readExpr(t));
		} while (t.try_match(","));
		t.match(")");
		return move(expr);
	}

	auto expr = std::make_unique<FieldNameExpr>(name);

	if (t.try_match("["))
	{
		expr->index = t.read<int>();
		t.match("]");
	}

	return move(expr);
};

auto readCondition(token_stream& t) -> Condition
{
	Condition c;
	c.field_name = t.readLabel();
	if (t.try_match("["))
	{
		c.index = t.read<int>();
		t.match("]");
	}

	if (t.try_match("="))
		c.op = Condition::E;
	else if (t.try_match("<"))
		c.op = Condition::LT;
	else if (t.try_match(">"))
		c.op = Condition::GT;
	else if (t.try_match("<="))
		c.op = Condition::LTE;
	else if (t.try_match(">="))
		c.op = Condition::GTE;
	else if (t.try_match("<>"))
		c.op = Condition::NE;
	else if (t.try_match("IN"))
		c.op = Condition::IN;
	else
		throw std::invalid_argument("CubeSQL::parse");

	if (t.is<AnyAtom>())
		c.a = t.read<AnyAtom>();
	else if (t.is<AnySet>())
		c.s = t.read<AnySet>();
	else if (t.is<AnyRange>())
		c.r = t.read<AnyRange>();

	return c;
};

auto readOrderingTerm(token_stream& t) -> OrderingTerm
{
	auto expr = readExpr(t);
	auto asc = true;
	if (t.try_match("ASC"))
		asc = true;
	else if (t.try_match("DESC"))
		asc = false;

	return OrderingTerm{std::move(expr), asc};
};

auto parse(const std::vector<token>& data) -> Select
{
	Select s;

	auto t = token_stream{data};

	t.match("SELECT");
	s.select.push_back(readExpr(t));
	while (t.try_match(","))
		s.select.push_back(readExpr(t));

	if (t.try_match("WHERE"))
	{
		s.where.push_back(readCondition(t));
		while (t.try_match("AND"))
			s.where.push_back(readCondition(t));
	}

	if (t.try_match("ORDER"))
	{
		t.match("BY");
		s.order_by.push_back(readOrderingTerm(t));
		while (t.try_match(","))
			s.order_by.push_back(readOrderingTerm(t));
	}

	if (t.try_match("LIMIT"))
	{
		s.limit = t.read<int>();
		if (t.try_match("OFFSET"))
			s.offset = t.read<int>();
		else if (t.try_match(","))
		{
			// Reverse them, as in SQLite syntax.
			s.offset = *s.limit;
			s.limit = t.read<int>();
		}
	}

	t.match_end();

	return s;
}

auto colType(std::string type) -> ColType
{
	if (iequals(type, "INT"))
		return ColType::Int;
	if (iequals(type, "FLOAT"))
		return ColType::Float;
	if (iequals(type, "TIME"))
		return ColType::Time;
	if (iequals(type, "TEXT"))
		return ColType::Text;
	if (iequals(type, "CHAR"))
		return ColType::Char;
}

auto readColDef(token_stream& t) -> ColDef
{
	auto def = ColDef{t.readLabel()};
	if (t.is<AnySet>())
	{
		def.s = t.read<AnySet>();
		def.type = ColType::Set;
	}
	else if (t.is<AnyRange>())
	{
		def.r = t.read<AnyRange>();
		if (def.r.type == Type::Int)
			def.type = ColType::IntRange;
		else if (def.r.type == Type::Float)
			def.type = ColType::FloatRange;
		else
			throw std::invalid_argument("CubeSQL::parse");
	}
	else
		def.type = colType(t.readLabel());

	if (t.try_match("["))
	{
		def.len = t.read<int>();
		t.match("]");
	}

	return def;
}

auto parseCubeDef(const std::vector<token>& data) -> CubeDef
{
	auto t = token_stream{data};

	auto def = CubeDef{};

	while (t.try_match("DIM") || t.try_match("DIMENSION"))
	{
		def.dims.push_back(readColDef(t));

		if (!t.is_end())
			t.match(",");
	}

	while (t.try_match("MEA") || t.try_match("MEASURE"))
	{
		def.meas.push_back(readColDef(t));

		if (!t.is_end())
			t.match(",");
	}

	return def;
}

}

#include "tokenizer.h"

namespace CubeSQL {

auto parse(const std::string& data) -> Select
{
	return parse(tokenize(data));
}

auto parseCubeDef(const std::string& data) -> CubeDef
{
	return parseCubeDef(tokenize(data));
}

}
