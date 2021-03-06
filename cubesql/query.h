#pragma once

#include <memory>
#include <vector>
#include <cxxcompat/optional>
#include "types.h"

namespace CubeSQL
{
	struct Expr
	{
		virtual ~Expr();
	};

	using AnyExpr = std::unique_ptr<Expr>;

	struct FieldNameExpr: public Expr
	{
		std::string name;
		std::optional<int> index;
		FieldNameExpr(std::string&& n): name(std::move(n)) {}
		FieldNameExpr(const std::string& n): name(n) {}
	};

	struct ConstantExpr: public Expr
	{
		AnyAtom val;
		ConstantExpr(AnyAtom&& v): val(std::move(v)) {}
		ConstantExpr(const AnyAtom& v): val(v) {}
	};

	struct OperationExpr : public Expr
	{
		std::string name;
		std::vector<AnyExpr> args;
		OperationExpr(std::string&& n): name(std::move(n)) {}
		OperationExpr(const std::string& n): name(n) {}
	};

	struct Condition
	{
		std::string field_name;
		std::optional<int> index;
		enum Operator
		{
			E,
			LT,
			GT,
			LTE,
			GTE,
			NE,
			IN
		} op;

		AnyAtom  a;
		AnyRange r;
		AnySet   s;
	};

	struct OrderingTerm
	{
		AnyExpr term;
		bool asc = true;

		OrderingTerm(AnyExpr term, bool asc = true): term(std::move(term)), asc(asc) {}
	};

	struct Select
	{
		std::vector<AnyExpr> select;
		std::vector<Condition> where;
		std::vector<OrderingTerm> order_by;
		std::optional<int> limit;
		int offset = 0;
	};
}
