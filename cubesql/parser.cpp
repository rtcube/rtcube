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
auto token_stream::is<int>() -> bool
{
	return tokens[i].val.type == AnyAtom::Int;
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
		c.array_specifier = t.read<int>();
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
	//else if (t.is("{"))
	//	c.s = t.read<AnySet>();
	//else if (t.is("(") || t.is("<"))
	//	c.r = t.read<AnyRange>();

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

}
